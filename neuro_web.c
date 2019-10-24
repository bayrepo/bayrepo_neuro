/*
 * neuro_web.c
 *
 *  Created on: Oct 24, 2019
 *      Author: alexey
 */

#include <stdio.h>
#include <unistd.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>

#include "civetweb.h"
#include "neuro.h"

#define MAX_ACT_LAYERS 4096
#define MAX_PAGE_LEN 1024*1024

#define NOT_ENOUGH_MEMORY(x) if (!x) { \
		fprintf(stderr, "Not enough memory. Terminated\n"); \
		exit(255); \
	}

volatile int globalIdCounter = 1;

typedef struct __neuro_item {
	int id;
	char *name;
	void *net;
	int inputs;
	int outputs;
	int hiddens;
	int hid_nets;
	int dropout;
	activation act;
	activation flist[MAX_ACT_LAYERS];
	struct __neuro_item *next;
} neuro_item;

typedef struct __AddHandlerForm {
	int inputs;
	int outputs;
	int hidd;
	int hid_nets;
	int dropout;
	char *netName;
	activation act;
	activation flist[MAX_ACT_LAYERS];
	struct mg_connection *conn;
} AddHandlerForm;

neuro_item *neuro_head = NULL;
neuro_item *neuro_tail = NULL;

pthread_rwlock_t m_globalNeuro = PTHREAD_RWLOCK_INITIALIZER;

//Common fucntions
static int web_add_new_net(AddHandlerForm *form, char *error_buffer) {
	int index;
	neuro_item *item = calloc(1, sizeof(neuro_item));
	NOT_ENOUGH_MEMORY(item);
	item->inputs = form->inputs;
	item->outputs = form->outputs;
	item->hiddens = form->hidd;
	item->hid_nets = form->hid_nets;
	item->dropout = form->dropout;
	item->name = form->netName;
	item->act = form->act;
	for (index = 0; index < MAX_ACT_LAYERS; index++) {
		item->flist[index] = form->flist[index];
	}
	item->net = bayrepo_init_neuro(3, 1, 4, 1, 0.2, RELU);
	NOT_ENOUGH_MEMORY(item->net);
	pthread_rwlock_wrlock(&m_globalNeuro);
	item->id = globalIdCounter++;
	if (neuro_head) {
		neuro_tail->next = item;
		neuro_tail = item;
	} else {
		neuro_head = item;
		neuro_tail = item;
	}
	pthread_rwlock_unlock(&m_globalNeuro);
	return 0;
}

static void web_delete_net(int id) {

}

static neuro_item *web_get_neuro(int id) {
	neuro_item *item = neuro_head;

	if (item) {
		while (item) {
			if (item->id == id)
				return item;
			item = item->next;
		}
	}
	return NULL;
}

//Web interface functions
#define DOCUMENT_ROOT "."

#define PORT "10111"

#define EXIT_URI "/exit"
#define ADD_URI "/add"
#define GET_URI "/get"
#define IMG_URI "/img"
#define ADD_FILE_URI "/addf"
volatile int exitNow = 0;

static int web_IndexHandler(struct mg_connection *conn, void *cbdata) {
	mg_send_http_ok(conn, "text/html", MAX_PAGE_LEN);

	mg_printf(conn, "<!DOCTYPE html>\n");
	mg_printf(conn, "<html>\n");
	mg_printf(conn, "  <head>\n");
	mg_printf(conn, "    <meta charset=\"utf-8\">\n");
	mg_printf(conn,
			"    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
	mg_printf(conn, "    <title>Neuro creator</title>\n");
	mg_printf(conn, "    <style>\n");
	mg_printf(conn, "      .tbl1{\n");
	mg_printf(conn, "        width: 100%;\n");
	mg_printf(conn, "      }\n");
	mg_printf(conn, "      .neuroname{\n");
	mg_printf(conn, "        padding: 4px;\n");
	mg_printf(conn, "      }\n");
	mg_printf(conn, "    </style>\n");
	mg_printf(conn, "  </head>\n");
	mg_printf(conn, "  <body>\n");
	mg_printf(conn, "    <h1>List of nets</h1>\n");
	mg_printf(conn, "    <table class=\"tbl1\">\n");
	mg_printf(conn, "      <tr>\n");
	mg_printf(conn, "        <th><a href=\"%s\">Add neuro</a></th>\n", ADD_URI);
	mg_printf(conn, "        <th><a href=\"%s\">Add neuro from file</a></th>\n",
	ADD_FILE_URI);
	mg_printf(conn, "        <th><a href=\"%s\">Exit</a></th>\n", EXIT_URI);
	mg_printf(conn, "      </tr>\n");
	pthread_rwlock_rdlock(&m_globalNeuro);
	neuro_item *item = neuro_head;
	mg_printf(conn, "      <tr>\n");
	mg_printf(conn, "        <td colspan=\"3\">\n");
	if (item) {
		while (item) {
			mg_printf(conn,
					"      <span class=\"neuroname\"><a href=\"%s/%d\">%s</a></span>\n",
					GET_URI, item->id, item->name);
			item = item->next;
		}
	} else {
		mg_printf(conn, "&nbsp;\n");
	}
	pthread_rwlock_unlock(&m_globalNeuro);
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "      </tr>\n");
	mg_printf(conn, "    </table>\n");
	mg_printf(conn, "  </body>\n");
	mg_printf(conn, "</html>\n");

	return 1;
}

static int web_field_found_AddForm(const char *key, const char *filename,
		char *path, size_t pathlen, void *user_data) {
	return MG_FORM_FIELD_STORAGE_GET;
}

static int web_field_get_AddForm(const char *key, const char *value,
		size_t valuelen, void *user_data) {
	AddHandlerForm *form = (AddHandlerForm *) user_data;
	if ((key != NULL) && (key[0] == '\0')) {
		return MG_FORM_FIELD_HANDLE_ABORT;
	}
	if ((valuelen > 0) && (value == NULL)) {
		return MG_FORM_FIELD_HANDLE_ABORT;
	}

	if (key) {
		if (!strcmp(key, "netname")) {
			form->netName =
					value[0] ? strndup(value, valuelen) : strdup("defnet");
			NOT_ENOUGH_MEMORY(form->netName);
		} else if (!strcmp(key, "inputs")) {
			int vl = 0;
			vl = atoi(value);
			form->inputs = (vl > 0) ? vl : 1;
		} else if (!strcmp(key, "outputs")) {
			int vl = 0;
			vl = atoi(value);
			form->outputs = (vl > 0) ? vl : 1;
		} else if (!strcmp(key, "hidd")) {
			int vl = 0;
			vl = atoi(value);
			form->hidd = (vl > 0) ? vl : 0;
		} else if (!strcmp(key, "hiddnets")) {
			int vl = 0;
			vl = atoi(value);
			form->hid_nets = (vl > 0) ? vl : 0;
		} else if (!strcmp(key, "activg")) {
			int vl = atoi(value);
			switch (vl) {
			case 0:
				form->act = SIGMOID;
				break;
			case 1:
				form->act = TANH;
				break;
			case 2:
				form->act = RELU;
				break;
			case 3:
				form->act = NOACTIV;
				break;
			default:
				form->act = DEFLT;
				break;
			}
		} else if (strstr(key, "activl_")) {
			const char *p = key + strlen("activl_");
			int kn = atoi(p);
			int vl = atoi(value);
			if (kn < MAX_ACT_LAYERS) {
				switch (vl) {
				case 0:
					form->flist[kn] = SIGMOID;
					break;
				case 1:
					form->flist[kn] = TANH;
					break;
				case 2:
					form->flist[kn] = RELU;
					break;
				case 3:
					form->flist[kn] = NOACTIV;
					break;
				default:
					form->flist[kn] = DEFLT;
					break;
				}
			}
		} else if (!strcmp(key, "drp")) {
			form->dropout = 1;
		}
	}

	return 0;
}

static int web_field_stored_AddForm(const char *path, long long file_size,
		void *user_data) {
	return 0;
}

static int web_AddHandler(struct mg_connection *conn, void *cbdata) {
	const struct mg_request_info *req_info = mg_get_request_info(conn);
	int ret, ret2;
	char result_error[MAX_ACT_LAYERS] = "";
	AddHandlerForm form_container;
	memset(&form_container, 0, sizeof(AddHandlerForm));
	int index;
	for (index = 0; index < MAX_ACT_LAYERS; index++) {
		form_container.flist[index] = DEFLT;
	}
	form_container.conn = conn;
	struct mg_form_data_handler fdh = { web_field_found_AddForm,
			web_field_get_AddForm, web_field_stored_AddForm, &form_container };

	ret = mg_handle_form_request(conn, &fdh);
	if (ret > 0) {
		ret2 = web_add_new_net(&form_container, result_error);
		if (!ret2) {
			mg_send_http_redirect(conn, "/", 302);
			return 1;
		}
	}

	mg_send_http_ok(conn, "text/html", MAX_PAGE_LEN);
	mg_printf(conn, "<!DOCTYPE html>\n");
	mg_printf(conn, "<html>\n");
	mg_printf(conn, "  <head>\n");
	mg_printf(conn, "    <meta charset=\"utf-8\">\n");
	mg_printf(conn,
			"    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
	mg_printf(conn, "    <title>Neuro creator(add net)</title>\n");
	mg_printf(conn, "    <script>\n");
	mg_printf(conn, "      function addlayers(val){\n");
	mg_printf(conn,
			"        elem = document.getElementById('layercontainer');\n");
	mg_printf(conn, "        ln = parseInt(val) + 1;\n");
	mg_printf(conn, "        bd = \"\";\n");
	mg_printf(conn, "        fst = 1\n");
	mg_printf(conn, "        while(ln>0){\n");
	mg_printf(conn, "          lnames = \"Layer func \"+ ln.toString();\n");
	mg_printf(conn, "          if (fst == 1) {\n");
	mg_printf(conn, "            fst = 0;\n");
	mg_printf(conn, "            lnames = \"Output layer\";\n");
	mg_printf(conn, "          }\n");
	mg_printf(conn,
			"          tt=\"<label for=\\\"func-name-\"+ ln.toString() +\"\\\">\"+ lnames +\":</label><select id=\\\"func-name-\"+ ln.toString() +\"\\\" name=\\\"activl_\"+ ln.toString()+\"\\\"><option selected value=\\\"-1\\\">NODEF</option><option value=\\\"0\\\">SIGMOID</option><option value=\\\"1\\\">TANH</option><option value=\\\"2\\\">RELU</option><option value=\\\"3\\\">NO ACTIVATION</option></select><br>\";\n");
	mg_printf(conn, "          bd = tt + bd;\n");
	mg_printf(conn, "          ln--;\n");
	mg_printf(conn, "        }\n");
	mg_printf(conn, "        elem.innerHTML = bd;\n");
	mg_printf(conn, "      }\n");
	mg_printf(conn, "    </script>\n");
	mg_printf(conn, "  </head>\n");
	mg_printf(conn, "  <body>\n");
	mg_printf(conn, "    <h1>Add net</h1>\n");
	if (ret > 0) {
		mg_printf(conn, "   <p>%s<p>\n", result_error);
	}
	mg_printf(conn,
			"    <form action=\"/add\" method=\"POST\" enctype=\"multipart/form-data\">\n");
	mg_printf(conn, "      <label for=\"net-name\">Name</label>\n");
	mg_printf(conn,
			"      <input id=\"net-name\" type=\"text\" name=\"netname\"><br>\n");
	mg_printf(conn, "      <label for=\"inp-name\">Inputs</label>\n");
	mg_printf(conn,
			"      <input id=\"inp-name\" type=\"text\" name=\"inputs\"><br>\n");
	mg_printf(conn, "      <label for=\"out-name\">Outputs</label>\n");
	mg_printf(conn,
			"      <input id=\"out-name\" type=\"text\" name=\"outputs\"><br>\n");
	mg_printf(conn, "      <label for=\"hid-name\">Hidden</label>\n");
	mg_printf(conn,
			"      <input id=\"hid-name\" type=\"text\" name=\"hidd\" onchange=\"addlayers(this.value)\"><br>\n");
	mg_printf(conn, "      <label for=\"hidn-name\">Nets of hidden</label>\n");
	mg_printf(conn,
			"      <input id=\"hidn-name\" type=\"text\" name=\"hiddnets\"><br>\n");
	mg_printf(conn,
			"      <label for=\"func-name\">Activation func:</label>\n");
	mg_printf(conn, "      <select id=\"func-name\" name=\"activg\">\n");
	mg_printf(conn, "        <option selected value=\"0\">SIGMOID</option>\n");
	mg_printf(conn, "        <option value=\"1\">TANH</option>\n");
	mg_printf(conn, "        <option value=\"2\">RELU</option>\n");
	mg_printf(conn, "        <option value=\"3\">NO ACTIVATION</option>\n");
	mg_printf(conn, "      </select><br>\n");
	mg_printf(conn, "      <label for=\"drp-name\">Dropout:</label>\n");
	mg_printf(conn,
			"      <input id=\"drp-name\" type=\"checkbox\" name=\"drp\" value=\"y\"><br>\n");
	mg_printf(conn, "      <div id=\"layercontainer\">\n");
	mg_printf(conn, "      </div>\n");
	mg_printf(conn, "\n");
	mg_printf(conn, "      <input type=\"submit\" value=\"Create\">\n");
	mg_printf(conn, "    </form>\n");
	mg_printf(conn, "\n");
	mg_printf(conn, "  </body>\n");
	mg_printf(conn, "</html>\n");
	return 1;
}

static int web_ExitHandler(struct mg_connection *conn, void *cbdata) {
	pthread_rwlock_rdlock(&m_globalNeuro);
	neuro_item *item = neuro_head;

	if (item) {
		while (item) {
			free(item->name);
			bayrepo_clean_neuro(item->net);
			neuro_item *item_old = item;
			item = item->next;
			free(item_old);
		}
	}

	pthread_rwlock_unlock(&m_globalNeuro);
	mg_send_http_ok(conn, "text/plain", MAX_PAGE_LEN);
	mg_printf(conn, "Server will shut down.\n");
	mg_printf(conn, "Bye!\n");
	exitNow = 1;
	return 1;
}

static const char *web_translate_active_to_string(activation act) {
	switch (act) {
	case SIGMOID:
		return "SIGMOID";
	case TANH:
		return "TANH";
	case RELU:
		return "RELU";
	case NOACTIV:
		return "NO ACTIVATION";
	default:
		return "NO DEFINED";
	}
}

static int web_GetHandler(struct mg_connection *conn, void *cbdata) {
	const struct mg_request_info *req_info = mg_get_request_info(conn);
	int id = 0;
	int index = 0;

	char *uri = strdup(req_info->request_uri);
	NOT_ENOUGH_MEMORY(uri);

	char *dash = strchr(uri + 1, '/');
	if (dash) {
		char *secondDash = strchr(dash + 1, '/');
		if (secondDash) {
			*secondDash = 0;
		}
		id = atoi(dash + 1);
	}

	pthread_rwlock_rdlock(&m_globalNeuro);
	neuro_item *item = web_get_neuro(id);
	if (!item) {
		pthread_rwlock_unlock(&m_globalNeuro);
		mg_send_http_error(conn, 404, "Not Found");
		return 0;
	}
	mg_send_http_ok(conn, "text/html", MAX_PAGE_LEN);

	mg_printf(conn, "<!DOCTYPE html>\n");
	mg_printf(conn, "<html>\n");
	mg_printf(conn, "  <head>\n");
	mg_printf(conn, "    <meta charset=\"utf-8\">\n");
	mg_printf(conn,
			"    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
	mg_printf(conn, "    <title>Neuro creator(net1)</title>\n");
	mg_printf(conn, "    <style>\n");
	mg_printf(conn, "      .tbl1{\n");
	mg_printf(conn, "        width: 100%;\n");
	mg_printf(conn, "      }\n");
	mg_printf(conn, "      \n");
	mg_printf(conn, "      .tbl1 td{\n");
	mg_printf(conn, "        vertical-align: top\n");
	mg_printf(conn, "      }\n");
	mg_printf(conn, "      \n");
	mg_printf(conn, "      .tbl2{\n");
	mg_printf(conn, "        width: 100%;\n");
	mg_printf(conn, "      }\n");
	mg_printf(conn, "      .tbl2 td{\n");
	mg_printf(conn, "        border: 1px solid #c71818;\n");
	mg_printf(conn, "        text-align: center\n");
	mg_printf(conn, "      }\n");
	mg_printf(conn, "      .tbl2 th{\n");
	mg_printf(conn, "        background-color: #c8e0e0;\n");
	mg_printf(conn, "        text-align: center;\n");
	mg_printf(conn, "      }\n");
	mg_printf(conn, "    </style>\n");
	mg_printf(conn, "  </head>\n");
	mg_printf(conn, "  <body>\n");
	mg_printf(conn, "    <h1>%s</h1>\n", item->name);
	mg_printf(conn, "    <table class=\"tbl1\">\n");
	mg_printf(conn, "      <tr>\n");
	mg_printf(conn, "        <td>\n");
	mg_printf(conn, "          <table>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <th>Parameter</th>\n");
	mg_printf(conn, "              <th>Value</th>\n");
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>Net name</td>\n");
	mg_printf(conn, "              <td>%s</td>\n", item->name);
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>Inputs</td>\n");
	mg_printf(conn, "              <td>%d</td>\n", item->inputs);
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>Outputs</td>\n");
	mg_printf(conn, "              <td>%d</td>\n", item->outputs);
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>Hidden</td>\n");
	mg_printf(conn, "              <td>%d</td>\n", item->hiddens);
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>Neurons in hidden layer</td>\n");
	mg_printf(conn, "              <td>%d</td>\n", item->hid_nets);
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>Activation function</td>\n");
	mg_printf(conn, "              <td>%s</td>\n",
			web_translate_active_to_string(item->act));
	mg_printf(conn, "            </tr>\n");
	for (index = 0; index < (item->hiddens + 1); index++) {
		mg_printf(conn, "            <tr>\n");
		if (index == item->hiddens) {
			mg_printf(conn, "              <td>Activation func output</td>\n");
		} else {
			mg_printf(conn, "              <td>Activation func layer 1</td>\n");
		}
		mg_printf(conn, "              <td>%s</td>\n",
				web_translate_active_to_string(item->flist[index]));
		mg_printf(conn, "            </tr>\n");
	}
	mg_printf(conn, "          </table>\n");
	mg_printf(conn, "\n");
	mg_printf(conn, "        </td>\n");
	mg_printf(conn,
			"        <td><img src=\"%s/%d\"></td>\n", IMG_URI, id);
	mg_printf(conn, "      </tr>\n");
	mg_printf(conn, "      <tr>\n");
	mg_printf(conn, "        <td>\n");
	mg_printf(conn, "          <form action=\"%s/%d\" method=\"POST\">\n", GET_URI, id);
	mg_printf(conn,
			"            <label for=\"file_in1\">Input parameters file</label>\n");
	mg_printf(conn,
			"            <input id=\"file_in1\" type=\"file\" name=\"fin1\"><br>\n");
	mg_printf(conn, "            <input type=\"submit\" value=\"Send\">\n");
	mg_printf(conn, "          </form>\n");
	mg_printf(conn, "          <form action=\"%s/%d\" method=\"POST\">\n", GET_URI, id);
	mg_printf(conn,
			"            <label for=\"file_in2_1\">Input parameter 1</label>\n");
	mg_printf(conn,
			"            <input id=\"file_in2_1\" type=\"text\" name=\"fin2_1\"><br>\n");
	mg_printf(conn,
			"            <label for=\"file_in2_2\">Input parameter 2</label>\n");
	mg_printf(conn,
			"            <input id=\"file_in2_2\" type=\"text\" name=\"fin2_2\"><br>\n");
	mg_printf(conn, "            <input type=\"submit\" value=\"Send\">\n");
	mg_printf(conn, "          </form>\n");
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "        <td>\n");
	mg_printf(conn, "          <table>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <th>Result output</th>\n");
	mg_printf(conn, "              <th>Result value</th>\n");
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>\n");
	mg_printf(conn, "                1\n");
	mg_printf(conn, "              </td>\n");
	mg_printf(conn, "              <td>\n");
	mg_printf(conn, "                0.566\n");
	mg_printf(conn, "              </td>\n");
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "          </table>\n");
	mg_printf(conn, "\n");
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "      </tr>\n");
	mg_printf(conn, "      <tr>\n");
	mg_printf(conn, "        <td>\n");
	mg_printf(conn, "          <form action=\"%s/%d\" method=\"POST\">\n", GET_URI, id);
	mg_printf(conn,
			"            <label for=\"file_train1\">Input training parameters file</label>\n");
	mg_printf(conn,
			"            <input id=\"file_train1\" type=\"file\" name=\"tin1\"><br>\n");
	mg_printf(conn, "            <input type=\"submit\" value=\"Send\">\n");
	mg_printf(conn, "          </form>\n");
	mg_printf(conn, "          <form action=\"%s/%d\" method=\"POST\">\n", GET_URI, id);
	mg_printf(conn,
			"            <label for=\"file_train2_1\">Input parameter train 1</label>\n");
	mg_printf(conn,
			"            <input id=\"file_train2_1\" type=\"text\" name=\"tin2_1\"><br>\n");
	mg_printf(conn,
			"            <label for=\"file_train_exp2_1\">Result expected 1</label>\n");
	mg_printf(conn,
			"            <input id=\"file_train_exp2_1\" type=\"text\" name=\"tine2_1\"><br>\n");
	mg_printf(conn, "            <input type=\"submit\" value=\"Send\">\n");
	mg_printf(conn, "          </form>\n");
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "        <td>\n");
	mg_printf(conn, "          <table class=\"tbl2\">\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <th colspan=\"2\">Layer 0</th>\n");
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>0.6765</td>\n");
	mg_printf(conn, "              <td>0.778778</td>\n");
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "          </table>\n");
	mg_printf(conn, "          <table class=\"tbl2\">\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <th colspan=\"4\">Layer 0</th>\n");
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <td>0.6765</td>\n");
	mg_printf(conn, "              <td>0.778778</td>\n");
	mg_printf(conn, "              <td>0.6765</td>\n");
	mg_printf(conn, "              <td>0.778778</td>\n");
	mg_printf(conn, "            </tr>\n");
	mg_printf(conn, "          </table>\n");
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "      </tr>\n");
	mg_printf(conn, "    </table>\n");

	pthread_rwlock_unlock(&m_globalNeuro);

	return 1;
}

static int web_log_message(const struct mg_connection *conn,
		const char *message) {
	puts(message);
	return 1;
}

int main(int argc, char *argv[]) {
	const char *options[] = {
#if !defined(NO_FILES)
			"document_root",
			DOCUMENT_ROOT,
#endif
			"listening_ports",
			PORT, "request_timeout_ms", "10000", "error_log_file", "error.log",
			0 };
	struct mg_callbacks callbacks;
	struct mg_context *ctx;
	struct mg_server_port ports[32];
	int port_cnt, n;
	int err = 0;

	if (err) {
		fprintf(stderr, "Cannot start neuroweb - inconsistent build.\n");
		return EXIT_FAILURE;
	}

	memset(&callbacks, 0, sizeof(callbacks));
	callbacks.log_message = web_log_message;
	ctx = mg_start(&callbacks, 0, options);

	if (ctx == NULL) {
		fprintf(stderr, "Cannot start neuroweb - mg_start failed.\n");
		return EXIT_FAILURE;
	}

	mg_set_request_handler(ctx, EXIT_URI, web_ExitHandler, 0);
	mg_set_request_handler(ctx, ADD_URI, web_AddHandler, 0);
	mg_set_request_handler(ctx, GET_URI, web_GetHandler, 0);
	//mg_set_request_handler(ctx, IMG_URI, web_ImageHandler, 0);
	mg_set_request_handler(ctx, "/", web_IndexHandler, 0);

	memset(ports, 0, sizeof(ports));
	port_cnt = mg_get_server_ports(ctx, 32, ports);
	printf("\n%i listening ports:\n\n", port_cnt);

	for (n = 0; n < port_cnt && n < 32; n++) {
		const char *proto = ports[n].is_ssl ? "https" : "http";
		const char *host;

		if ((ports[n].protocol & 1) == 1) {
			/* IPv4 */
			host = "127.0.0.1";
			printf("View neuro at %s://%s:%i/\n", proto, host, ports[n].port);
			printf("Exit at %s://%s:%i%s\n", proto, host, ports[n].port,
			EXIT_URI);
			printf("\n");
		}

	}

	/* Wait until the server should be closed */
	while (!exitNow) {
		sleep(1);
	}

	/* Stop the server */
	mg_stop(ctx);
	printf("Server stopped.\n");
	printf("Bye!\n");

	return EXIT_SUCCESS;
}

