/*
 * neuro_web.c
 *
 *  Created on: Oct 24, 2019
 *      Author: alexey
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <unistd.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>

#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>

#include "civetweb.h"
#include "neuro.h"
#include "neuro_urls.h"

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
	double learn_speed;
	activation act;
	activation flist[MAX_ACT_LAYERS];
	struct __neuro_item *next;
	struct __neuro_item *prev;
} neuro_item;

typedef struct __AddHandlerForm {
	int inputs;
	int outputs;
	int hidd;
	int hid_nets;
	int dropout;
	char *netName;
	double learn_spd;
	activation act;
	activation flist[MAX_ACT_LAYERS];
	struct mg_connection *conn;
} AddHandlerForm;

neuro_item *neuro_head = NULL;
neuro_item *neuro_tail = NULL;

pthread_rwlock_t m_globalNeuro = PTHREAD_RWLOCK_INITIALIZER;

//Common fucntions
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
	item->learn_speed = form->learn_spd;
	for (index = 0; index < MAX_ACT_LAYERS; index++) {
		item->flist[index] = form->flist[index];
	}
	item->net = bayrepo_init_neuro(item->inputs, item->outputs, item->hid_nets,
			item->hiddens, item->learn_speed, item->act);
	NOT_ENOUGH_MEMORY(item->net);
	for (index = 0; index < (item->hiddens + 1); index++) {
		bayrepo_set_layer_activ(item->net, index, item->flist[index]);
	}
	pthread_rwlock_wrlock(&m_globalNeuro);
	item->id = globalIdCounter++;
	if (neuro_head) {
		neuro_tail->next = item;
		item->prev = neuro_tail;
		neuro_tail = item;
	} else {
		neuro_head = item;
		neuro_tail = item;
	}
	pthread_rwlock_unlock(&m_globalNeuro);
	return 0;
}

static void web_delete_net(int id) {
	pthread_rwlock_wrlock(&m_globalNeuro);
	neuro_item *item = web_get_neuro(id);
	if (item) {
		if (neuro_head == item) {
			neuro_head = item->next;
		}
		if (neuro_tail == item) {
			neuro_tail = item->prev;
		}
		if (item->prev) {
			item->prev->next = item->next;
		}
		if (item->next) {
			item->next->prev = item->prev;
		}
		if (item->name)
			free(item->name);
		if (item->net)
			bayrepo_clean_neuro(item->net);
		free(item);
	}
	pthread_rwlock_unlock(&m_globalNeuro);
}

//Web interface functions
#define DOCUMENT_ROOT "."

#define PORT "10111"

#define EXIT_URI "/exit"
#define ADD_URI "/addn"
#define GET_URI "/get"
#define ADD_FILE_URI "/addf"
#define DEL_URI "/del"
#define SVALL_URI "/allsave"

#define NEURO_FILE "neuro_web.dat"

#define ECHO_URI "/echo"
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
//	mg_printf(conn, "        <th><a href=\"%s\">Add neuro from file</a></th>\n",
//	ADD_FILE_URI);
	mg_printf(conn, "        <th><a href=\"%s\">Exit</a></th>\n", EXIT_URI);
	mg_printf(conn, "        <th><a href=\"%s\">Save all</a></th>\n",
	SVALL_URI);
	mg_printf(conn, "      </tr>\n");
	pthread_rwlock_rdlock(&m_globalNeuro);
	neuro_item *item = neuro_head;
	mg_printf(conn, "      <tr>\n");
	mg_printf(conn, "        <td colspan=\"4\">\n");
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

static void web_clean_AddHandlerForm(AddHandlerForm *form) {
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
		} else if (!strcmp(key, "lnspd")) {
			double vl = 0.0;
			char *ptr;
			vl = strtod(value, &ptr);
			form->learn_spd = (vl > 0.0) ? vl : 0.2;
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
		web_clean_AddHandlerForm(&form_container);
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
	mg_printf(conn, "        ln = parseInt(val);\n");
	mg_printf(conn, "        bd = \"\";\n");
	mg_printf(conn, "        fst = 1\n");
	mg_printf(conn, "        while(ln>=0){\n");
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
			"    <form action=\"%s\" method=\"POST\" enctype=\"multipart/form-data\">\n",
			ADD_URI);
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
	mg_printf(conn, "      <label for=\"learnspd\">Learn speed</label>\n");
	mg_printf(conn,
			"      <input id=\"learnspd\" type=\"text\" name=\"lnspd\"><br>\n");
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
	web_clean_AddHandlerForm(&form_container);
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

static void web_GetHandler_print_header(void *user_data, char *hdr, int col) {
	struct mg_connection *conn = (struct mg_connection *) user_data;
	mg_printf(conn, "          <table class=\"tbl2\">\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <th colspan=\"%d\">%s</th>\n", col, hdr);
	mg_printf(conn, "            </tr>\n");
}

static void web_GetHandler_print_footer(void *user_data) {
	struct mg_connection *conn = (struct mg_connection *) user_data;
	mg_printf(conn, "          </table>\n");
}

static void web_GetHandler_print_pret(void *user_data) {
	struct mg_connection *conn = (struct mg_connection *) user_data;
	mg_printf(conn, "            <tr>\n");
}

static void web_GetHandler_print_post(void *user_data) {
	struct mg_connection *conn = (struct mg_connection *) user_data;
	mg_printf(conn, "            </tr>\n");
}

static void web_GetHandler_print(void *user_data, double dt) {
	struct mg_connection *conn = (struct mg_connection *) user_data;
	mg_printf(conn, "              <td>%f</td>\n", dt);
}

static char *web_base64(const unsigned char *input, int length) {
	BIO *bmem, *b64;
	BUF_MEM *bptr;

	b64 = BIO_new(BIO_f_base64());
	bmem = BIO_new(BIO_s_mem());
	b64 = BIO_push(b64, bmem);
	BIO_write(b64, input, length);
	BIO_flush(b64);
	BIO_get_mem_ptr(b64, &bptr);

	char *buff = (char *) malloc(bptr->length);
	memcpy(buff, bptr->data, bptr->length - 1);
	buff[bptr->length - 1] = 0;

	BIO_free_all(b64);

	return buff;
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
		return 1;
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
	mg_printf(conn, "    <h1>%s[<a href=\"%s/%d\">x</a>]</h1>\n", item->name,
	DEL_URI, item->id);
	mg_printf(conn, "    <p><a href=\"/\">Back</a></p>\n");
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
			mg_printf(conn, "              <td>Activation func layer %d</td>\n",
					index);
		}
		mg_printf(conn, "              <td>%s</td>\n",
				web_translate_active_to_string(item->flist[index]));
		mg_printf(conn, "            </tr>\n");
	}
	mg_printf(conn, "          </table>\n");
	mg_printf(conn, "\n");
	mg_printf(conn, "        </td>\n");

	bayrepo_mem_encode pic;
	bayrepo_write_matrix(item->net, stdout, 200, 200, &pic);
	if (pic.size) {
		char *pic_buf = web_base64(pic.buffer, pic.size);
		NOT_ENOUGH_MEMORY(pic_buf);
		mg_printf(conn,
				"        <td><img src=\"data:image/png;base64,%s\"/></td>\n",
				pic_buf);
		free(pic_buf);
		free(pic.buffer);
	} else {
		mg_printf(conn, "        <td>NO PICTURE PREVIEW</td>\n");
	}

	mg_printf(conn, "      </tr>\n");
	mg_printf(conn, "      <tr>\n");
	mg_printf(conn, "        <td>\n");
	mg_printf(conn,
			"          <form action=\"%s/%d\" method=\"POST\" enctype=\"multipart/form-data\">\n",
			ADD_INPUT_FILE, id);
	mg_printf(conn,
			"            <label for=\"file_in1\">Input parameters file</label>\n");
	mg_printf(conn,
			"            <input id=\"file_in1\" type=\"file\" name=\"fin1\"><br>\n");
	mg_printf(conn, "            <input type=\"submit\" value=\"Send\">\n");
	mg_printf(conn, "          </form>\n");
	mg_printf(conn, "          <form action=\"%s/%d\" method=\"POST\">\n",
	ADD_INPUTP, id);
	for (index = 0; index < item->inputs; index++) {
		mg_printf(conn,
				"            <label for=\"file_in2_%d\">Input parameter %d</label>\n",
				index, index);
		mg_printf(conn,
				"            <input id=\"file_in2_%d\" type=\"text\" name=\"fin2_%d\"><br>\n",
				index, index);
	}
	mg_printf(conn, "            <input type=\"submit\" value=\"Send\">\n");
	mg_printf(conn, "          </form>\n");
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "        <td>\n");
	mg_printf(conn, "          <table>\n");
	mg_printf(conn, "            <tr>\n");
	mg_printf(conn, "              <th>Result output</th>\n");
	mg_printf(conn, "              <th>Result value</th>\n");
	mg_printf(conn, "            </tr>\n");
	for (index = 0; index < item->outputs; index++) {
		mg_printf(conn, "            <tr>\n");
		mg_printf(conn, "              <td>\n");
		mg_printf(conn, "                %d\n", index);
		mg_printf(conn, "              </td>\n");
		mg_printf(conn, "              <td>\n");
		mg_printf(conn, "                %f\n",
				bayrepo_get_result(item->net, index));
		mg_printf(conn, "              </td>\n");
		mg_printf(conn, "            </tr>\n");
	}
	mg_printf(conn, "          </table>\n");
	mg_printf(conn, "\n");
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "      </tr>\n");
	mg_printf(conn, "      <tr>\n");
	mg_printf(conn, "        <td>\n");
	mg_printf(conn,
			"          <form action=\"%s/%d\" method=\"POST\" enctype=\"multipart/form-data\">\n",
			ADD_TRAIN_FILE, id);
	mg_printf(conn,
			"            <label for=\"file_train1\">Input training parameters file</label>\n");
	mg_printf(conn,
			"            <input id=\"file_train1\" type=\"file\" name=\"tin1\"><br>\n");
	mg_printf(conn, "            <input type=\"submit\" value=\"Send\">\n");
	mg_printf(conn, "          </form>\n");
	mg_printf(conn, "          <form action=\"%s/%d\" method=\"POST\">\n",
	ADD_TRAINP, id);
	for (index = 0; index < item->inputs; index++) {
		mg_printf(conn,
				"            <label for=\"file_train2_%d\">Input parameter train %d</label>\n",
				index, index);
		mg_printf(conn,
				"            <input id=\"file_train2_%d\" type=\"text\" name=\"tin2_%d\"><br>\n",
				index, index);
	}
	for (index = 0; index < item->outputs; index++) {
		mg_printf(conn,
				"            <label for=\"file_train_exp2_%d\">Result expected %d</label>\n",
				index, index);
		mg_printf(conn,
				"            <input id=\"file_train_exp2_%d\" type=\"text\" name=\"tine2_%d\"><br>\n",
				index, index);
	}
	mg_printf(conn, "            <input type=\"submit\" value=\"Send\">\n");
	mg_printf(conn, "          </form>\n");
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "        <td>\n");
	bayrepo_decorator dc = { (void *) conn, web_GetHandler_print_header,
			web_GetHandler_print_footer, web_GetHandler_print_pret,
			web_GetHandler_print_post, web_GetHandler_print };
	bayrepo_print_matrix_custom(item->net, &dc);
	mg_printf(conn, "        </td>\n");
	mg_printf(conn, "      </tr>\n");
	mg_printf(conn, "    </table>\n");

	pthread_rwlock_unlock(&m_globalNeuro);

	return 1;
}

static int web_DelHandler(struct mg_connection *conn, void *cbdata) {
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

	web_delete_net(id);
	mg_send_http_redirect(conn, "/", 302);
	return 1;
}

typedef struct __FormItem {
	int number;
	double value;
	struct __FormItem *next;
} FormItem;

typedef struct __InputHandlerForm {
	FormItem *head;
	int ajax;
	FormItem *expect;
	FormItem *exp_head;
	struct mg_connection *conn;
} InputHandlerForm;

static void web_clear_InputHandlerForm(InputHandlerForm *form) {
	FormItem *item;
	FormItem *item_old;
	if (form->head) {
		item = form->head;
		while (item) {
			item_old = item;
			item = item->next;
			free(item_old);
		}
	}
	if (form->expect) {
		item = form->expect;
		while (item) {
			item_old = item;
			item = item->next;
			free(item_old);
		}
	}
	if (form->exp_head) {
		item = form->exp_head;
		while (item) {
			item_old = item;
			item = item->next;
			free(item_old);
		}
	}
}

static int web_field_found_InputForm(const char *key, const char *filename,
		char *path, size_t pathlen, void *user_data) {
	return MG_FORM_FIELD_STORAGE_GET;
}

static int web_field_get_InputForm(const char *key, const char *value,
		size_t valuelen, void *user_data) {
	InputHandlerForm *form = (InputHandlerForm *) user_data;
	if ((key != NULL) && (key[0] == '\0')) {
		return MG_FORM_FIELD_HANDLE_ABORT;
	}
	if ((valuelen > 0) && (value == NULL)) {
		return MG_FORM_FIELD_HANDLE_ABORT;
	}

	if (key) {
		if (!strcmp(key, "ajax")) {
			form->ajax = 1;
		} else if (strstr(key, "fin2_")) {
			const char *p = key + strlen("fin2_");
			int kn = atoi(p);
			char *ptr;
			double vl = strtod(value, &ptr);
			FormItem *item = calloc(1, sizeof(FormItem));
			NOT_ENOUGH_MEMORY(item);
			item->number = kn;
			item->value = vl;
			if (form->head) {
				FormItem *iterator = form->head;
				while (iterator->next) {
					iterator = iterator->next;
				}
				iterator->next = item;
			} else {
				form->head = item;
			}
		} else if (strstr(key, "tin2_")) {
			const char *p = key + strlen("tin2_");
			int kn = atoi(p);
			char *ptr;
			double vl = strtod(value, &ptr);
			FormItem *item = calloc(1, sizeof(FormItem));
			NOT_ENOUGH_MEMORY(item);
			item->number = kn;
			item->value = vl;
			if (form->exp_head) {
				FormItem *iterator = form->exp_head;
				while (iterator->next) {
					iterator = iterator->next;
				}
				iterator->next = item;
			} else {
				form->exp_head = item;
			}
		} else if (strstr(key, "tine2_")) {
			const char *p = key + strlen("tine2_");
			int kn = atoi(p);
			char *ptr;
			double vl = strtod(value, &ptr);
			FormItem *item = calloc(1, sizeof(FormItem));
			NOT_ENOUGH_MEMORY(item);
			item->number = kn;
			item->value = vl;
			if (form->expect) {
				FormItem *iterator = form->expect;
				while (iterator->next) {
					iterator = iterator->next;
				}
				iterator->next = item;
			} else {
				form->expect = item;
			}
		}
	}

	return 0;
}

static int web_field_stored_InputForm(const char *path, long long file_size,
		void *user_data) {
	return 0;
}

static int web_InputHandler(struct mg_connection *conn, void *cbdata) {
	const struct mg_request_info *req_info = mg_get_request_info(conn);
	int id = 0;
	int index = 0;
	char buf[MAX_ACT_LAYERS];

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

	InputHandlerForm form_container;
	memset(&form_container, 0, sizeof(InputHandlerForm));
	form_container.conn = conn;
	struct mg_form_data_handler fdh =
			{ web_field_found_InputForm, web_field_get_InputForm,
					web_field_stored_InputForm, &form_container };

	int ret = mg_handle_form_request(conn, &fdh);
	pthread_rwlock_wrlock(&m_globalNeuro);
	neuro_item *item = web_get_neuro(id);
	if (item) {
		snprintf(buf, MAX_ACT_LAYERS, "%s/%d", GET_URI, id);
		if (ret > 0) {
			if (form_container.head) {
				for (index = 0; index < item->inputs; index++) {
					bayrepo_fill_input(item->net, index, 0.0);
				}
				FormItem *iterator = form_container.head;
				while (iterator) {
					if (iterator->number < item->inputs) {
						bayrepo_fill_input(item->net, iterator->number,
								iterator->value);
					}
					iterator = iterator->next;
				}
				bayrepo_query(item->net);
			} else if (form_container.exp_head) {
				for (index = 0; index < item->inputs; index++) {
					bayrepo_fill_input(item->net, index, 0.0);
				}
				for (index = 0; index < item->outputs; index++) {
					bayrepo_fill_train(item->net, index, 0.0);
				}
				FormItem *iterator = form_container.exp_head;
				while (iterator) {
					if (iterator->number < item->inputs) {
						bayrepo_fill_input(item->net, iterator->number,
								iterator->value);
					}
					iterator = iterator->next;
				}
				iterator = form_container.expect;
				while (iterator) {
					if (iterator->number < item->outputs) {
						bayrepo_fill_train(item->net, iterator->number,
								iterator->value);
					}
					iterator = iterator->next;
				}
				bayrepo_train(item->net, 1, item->dropout);
			}
		}

	} else {
		snprintf(buf, MAX_ACT_LAYERS, "/");
	}

	if (form_container.ajax) {
		mg_send_http_ok(conn, "text/plain", MAX_PAGE_LEN);

		if (form_container.head) {
			mg_printf(conn, "TYPE:RESULT\n");
			if (item) {
				for (index = 0; index < item->outputs; index++) {
					mg_printf(conn, "%d:%f\n", index,
							bayrepo_get_result(item->net, index));
				}
			}
		} else if (form_container.exp_head) {
			mg_printf(conn, "TYPE:TRAINING\n");
		}

		web_clear_InputHandlerForm(&form_container);
		pthread_rwlock_unlock(&m_globalNeuro);
		return 1;
	}

	web_clear_InputHandlerForm(&form_container);
	pthread_rwlock_unlock(&m_globalNeuro);

	mg_send_http_redirect(conn, buf, 302);
	return 1;
}

typedef struct __FileHandlerForm {
	char *result;
	char *training;
	int size;
	int ajax;
	struct mg_connection *conn;
} FileHandlerForm;

static void web_clear_FileHandlerForm(FileHandlerForm *form) {
	if (form->result)
		free(form->result);
	if (form->training)
		free(form->training);
}

static int web_field_found_FileForm(const char *key, const char *filename,
		char *path, size_t pathlen, void *user_data) {
	return MG_FORM_FIELD_STORAGE_GET;
}

static int web_field_get_FileForm(const char *key, const char *value,
		size_t valuelen, void *user_data) {
	FileHandlerForm *form = (FileHandlerForm *) user_data;
	if ((key != NULL) && (key[0] == '\0')) {
		return MG_FORM_FIELD_HANDLE_ABORT;
	}
	if ((valuelen > 0) && (value == NULL)) {
		return MG_FORM_FIELD_HANDLE_ABORT;
	}

	if (key) {
		if (!strcmp(key, "ajax")) {
			form->ajax = 1;
		} else if (!strcmp(key, "fin1")) {
			form->result = calloc(1, valuelen);
			NOT_ENOUGH_MEMORY(form->result);
			memcpy(form->result, value, valuelen);
			form->size = valuelen;
		} else if (!strcmp(key, "tin1")) {
			form->training = calloc(1, valuelen);
			NOT_ENOUGH_MEMORY(form->training);
			memcpy(form->training, value, valuelen);
			form->size = valuelen;
		}
	}
	return 0;
}

static int web_field_stored_FileForm(const char *path, long long file_size,
		void *user_data) {
	return 0;
}

/*
 * RESULT FORMAT
 * [input pin]:[value double]\n
 * ...
 * [input pin]:[value double]\n
 *
 * TRAIN FORMAT
 * EPO:[number_of_sets]:[epoch training int]\n
 * INP:[input pin]:[value double]\n
 * ...
 * INP:[input pin]:[value double]\n
 * EXP:[output pib]:[value double]\n
 * ...
 * EXP:[output pib]:[value double]\n
 * END:0:0.0
 */
typedef struct __TrainingSet {
	int number_of_sets;
	int epochs;
	double *inputs;
	double *results;
} TrainingSet;

static int web_FileHandler(struct mg_connection *conn, void *cbdata) {
	char * line = NULL;
	ssize_t rd;
	size_t len = 0;
	const struct mg_request_info *req_info = mg_get_request_info(conn);

	int id = 0;
	int index = 0;
	char buf[MAX_ACT_LAYERS];

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

	FileHandlerForm form_container;
	memset(&form_container, 0, sizeof(FileHandlerForm));
	form_container.conn = conn;
	struct mg_form_data_handler fdh = { web_field_found_FileForm,
			web_field_get_FileForm, web_field_stored_FileForm, &form_container };

	int ret = mg_handle_form_request(conn, &fdh);
	pthread_rwlock_wrlock(&m_globalNeuro);
	neuro_item *item = web_get_neuro(id);
	if (item) {
		snprintf(buf, MAX_ACT_LAYERS, "%s/%d", GET_URI, id);
		if (ret > 0) {
			if (form_container.result) {
				FILE *fp = fmemopen((void*) form_container.result,
						form_container.size, "r");
				if (fp) {
					while ((rd = getline(&line, &len, fp)) != -1) {
						if (rd > 0) {
							char *ptr1 = line;
							char *ptr2 = strchr(ptr1, ':');
							if (ptr2) {
								*ptr2 = 0;
								ptr2++;
								int nmb = atoi(ptr1);
								char *ptr_tmp;
								double value = strtod(ptr2, &ptr_tmp);
								if (nmb < item->inputs) {
									bayrepo_fill_input(item->net, nmb, value);
								}
							}
						}
					}
					fclose(fp);
					if (line)
						free(line);
					line = NULL;
					bayrepo_query(item->net);
				}
			} else if (form_container.training) {
				FILE *fp = fmemopen((void*) form_container.training,
						form_container.size, "r");
				if (fp) {
					int ind = 0;
					TrainingSet ts = { 0 };
					while ((rd = getline(&line, &len, fp)) != -1) {
						if (rd > 0) {
							char *ptr1 = line;
							char *ptr2 = strchr(ptr1, ':');
							if (ptr2) {
								*ptr2 = 0;
								ptr2++;
								char *ptr3 = strchr(ptr2, ':');
								if (ptr3) {
									*ptr3 = 0;
									ptr3++;
									char *ptr4 = strchr(ptr3, '\n');
									if (ptr4)
										*ptr4 = 0;
									if (!strcmp(ptr1, "EPO")) {
										ts.number_of_sets = atoi(ptr2);
										ts.epochs = atoi(ptr3);
										ts.inputs = calloc(ts.number_of_sets,
												sizeof(double) * item->inputs);
										NOT_ENOUGH_MEMORY(ts.inputs);
										ts.results = calloc(ts.number_of_sets,
												sizeof(double) * item->outputs);
										NOT_ENOUGH_MEMORY(ts.results);
										ind = 0;
									} else if (!strcmp(ptr1, "END")) {
										ind++;
										if (ind >= ts.number_of_sets)
											break;
									} else if (!strcmp(ptr1, "INP")) {
										int nmb = atoi(ptr2);
										char *pptr;
										double val = strtod(ptr3, &pptr);
										if (nmb < item->inputs) {
											ts.inputs[(ind * item->inputs) + nmb] =
													val;
										}
									} else if (!strcmp(ptr1, "EXP")) {
										int nmb = atoi(ptr2);
										char *pptr;
										double val = strtod(ptr3, &pptr);
										if (nmb < item->outputs) {
											ts.results[(ind * item->outputs)
													+ nmb] = val;
										}
									}
								}
							}
						}
					}
					fclose(fp);
					if (line)
						free(line);
					line = NULL;
					int jnd = 0, xnd = 0;
					for (ind = 0; ind < ts.epochs; ind++) {
						for (jnd = 0; jnd < ts.number_of_sets; jnd++) {
							for (xnd = 0; xnd < item->inputs; xnd++) {
								bayrepo_fill_input(item->net, xnd,
										ts.inputs[(jnd * item->inputs) + xnd]);
							}
							for (xnd = 0; xnd < item->outputs; xnd++) {
								bayrepo_fill_train(item->net, xnd,
										ts.results[(jnd * item->outputs) + xnd]);
							}
							bayrepo_train(item->net, 1, item->dropout);
						}
					}
				}
			}
		}

	} else {
		snprintf(buf, MAX_ACT_LAYERS, "/");
	}

	if (form_container.ajax) {
		mg_send_http_ok(conn, "text/plain", MAX_PAGE_LEN);

		if (form_container.result) {
			mg_printf(conn, "TYPE:RESULT\n");
			if (item) {
				for (index = 0; index < item->outputs; index++) {
					mg_printf(conn, "%d:%f\n", index,
							bayrepo_get_result(item->net, index));
				}
			}
		} else if (form_container.training) {
			mg_printf(conn, "TYPE:TRAINING\n");
		}

		web_clear_FileHandlerForm(&form_container);
		pthread_rwlock_unlock(&m_globalNeuro);
		return 1;
	}

	web_clear_FileHandlerForm(&form_container);
	pthread_rwlock_unlock(&m_globalNeuro);

	mg_send_http_redirect(conn, buf, 302);
	return 1;
}

typedef struct __EchoHandlerForm {
	struct mg_connection *conn;
} EchoHandlerForm;

static int web_field_found_EchoForm(const char *key, const char *filename,
		char *path, size_t pathlen, void *user_data) {
	return MG_FORM_FIELD_STORAGE_GET;
}

static int web_field_get_EchoForm(const char *key, const char *value,
		size_t valuelen, void *user_data) {
	EchoHandlerForm *form = (EchoHandlerForm *) user_data;
	if ((key != NULL) && (key[0] == '\0')) {
		return MG_FORM_FIELD_HANDLE_ABORT;
	}
	if ((valuelen > 0) && (value == NULL)) {
		return MG_FORM_FIELD_HANDLE_ABORT;
	}

	if (key) {
		mg_printf(form->conn, "Found fieled %s=%s valuelen %d\n", key, value,
				valuelen);
	}
	return 0;
}

static int web_field_stored_EchoForm(const char *path, long long file_size,
		void *user_data) {
	return 0;
}

static int web_EchoHandler(struct mg_connection *conn, void *cbdata) {
	char * line = NULL;
	ssize_t rd;
	size_t len = 0;
	const struct mg_request_info *req_info = mg_get_request_info(conn);

	mg_send_http_ok(conn, "text/plain", MAX_PAGE_LEN);

	EchoHandlerForm form_container;
	memset(&form_container, 0, sizeof(EchoHandlerForm));
	form_container.conn = conn;
	struct mg_form_data_handler fdh = { web_field_found_EchoForm,
			web_field_get_EchoForm, web_field_stored_EchoForm, &form_container };

	int ret = mg_handle_form_request(conn, &fdh);
	mg_printf(conn, "Found %d form fields\n", ret);
	return 1;
}

static int web_log_message(const struct mg_connection *conn,
		const char *message) {
	puts(message);
	return 1;
}

/*
 * File format:
 * [DATA_BEG]
 * [int id]\n
 * [char name]\n;
 * [int inputs]\n
 * [int outputs]\n
 * [int hiddens]\n
 * [int hid_nets]\n
 * [int dropout]\n
 * [double learn_speed]\n
 * [int act]\n
 * [FORMAT][size]\n
 * [neuro format]
 * [DATA_END]
 */

static int web_SaveAllHandler(struct mg_connection *conn, void *cbdata) {
	char * line = NULL;
	ssize_t rd;
	size_t len = 0;
	const struct mg_request_info *req_info = mg_get_request_info(conn);
	pthread_rwlock_wrlock(&m_globalNeuro);
	neuro_item *item = neuro_head;

	if (item) {
		FILE *fp = fopen(NEURO_FILE, "w");
		if (fp) {
			while (item) {
				char *net_buffer = NULL;
				int len = bayrepo_save_to_buffer(item->net, &net_buffer);
				if (len > 0 && net_buffer) {
					fputs("[DATA_BEG]\n", fp);
					fprintf(fp, "%d\n", item->id);
					fprintf(fp, "%s\n", item->name);
					fprintf(fp, "%d\n", item->inputs);
					fprintf(fp, "%d\n", item->outputs);
					fprintf(fp, "%d\n", item->hiddens);
					fprintf(fp, "%d\n", item->hid_nets);
					fprintf(fp, "%d\n", item->dropout);
					fprintf(fp, "%f\n", item->learn_speed);
					fprintf(fp, "%d\n", (int) item->act);
					fprintf(fp, "[FORMAT]%d\n", len);
					fwrite(net_buffer, len, 1, fp);
					fputs("[DATA_END]\n", fp);
				}
				item = item->next;
			}
			fclose(fp);
		}
	}
	pthread_rwlock_unlock(&m_globalNeuro);
	mg_send_http_redirect(conn, "/", 302);
	return 1;
}

static void web_restore_nets_from_file() {
	char result_buf[MAX_ACT_LAYERS];
	FILE *fp = fopen(NEURO_FILE, "r");
	pthread_rwlock_wrlock(&m_globalNeuro);
	neuro_item *item = NULL;
	int is_data = 0;
	int cntr = 0;
	int index;
	if (fp) {
		while (!feof(fp)) {
			if (fgets(result_buf, MAX_ACT_LAYERS, fp)) {
				if (strstr(result_buf, "[DATA_BEG]")) {
					is_data = 1;
					item = calloc(1, sizeof(neuro_item));
					NOT_ENOUGH_MEMORY(item);
					if (neuro_head) {
						neuro_tail->next = item;
						item->prev = neuro_tail;
						neuro_tail = item;
					} else {
						neuro_head = item;
						neuro_tail = item;
					}
				} else if (strstr(result_buf, "[DATA_END]")) {
					is_data = 0;
					cntr = 0;
					for (index = 0; index < (item->hiddens + 1); index++) {
						item->flist[index] = bayrepo_get_sublayer_func(
								item->net, index);
					}
				} else if (is_data) {
					switch (cntr) {
					case 0: {
						result_buf[strlen(result_buf) - 1] = 0;
						item->id = atoi(result_buf);
						if (globalIdCounter < item->id) {
							globalIdCounter = item->id;
						}
						break;
					}
					case 1: {
						result_buf[strlen(result_buf) - 1] = 0;
						item->name = strdup(result_buf);
						NOT_ENOUGH_MEMORY(item->name);
						break;
					}
					case 2: {
						result_buf[strlen(result_buf) - 1] = 0;
						item->inputs = atoi(result_buf);
						break;
					}
					case 3: {
						result_buf[strlen(result_buf) - 1] = 0;
						item->outputs = atoi(result_buf);
						break;
					}
					case 4: {
						result_buf[strlen(result_buf) - 1] = 0;
						item->hiddens = atoi(result_buf);
						break;
					}
					case 5: {
						result_buf[strlen(result_buf) - 1] = 0;
						item->hid_nets = atoi(result_buf);
						break;
					}
					case 6: {
						result_buf[strlen(result_buf) - 1] = 0;
						item->dropout = atoi(result_buf);
						break;
					}
					case 7: {
						char *tptr;
						result_buf[strlen(result_buf) - 1] = 0;
						item->learn_speed = strtod(result_buf, &tptr);
						break;
					}
					case 8: {
						result_buf[strlen(result_buf) - 1] = 0;
						item->act = (activation) atoi(result_buf);
						is_data = 0;
						break;
					}
					}
					cntr++;
				} else if (strstr(result_buf, "[FORMAT]")) {
					result_buf[strlen(result_buf) - 1] = 0;
					char *ptr = result_buf;
					ptr += strlen("[FORMAT]");
					int sz = atoi(ptr);
					char *tmp_buffer = malloc(sz * sizeof(char));
					NOT_ENOUGH_MEMORY(tmp_buffer);
					fread(tmp_buffer, sz, 1, fp);
					item->net = bayrepo_restore_buffer(tmp_buffer, sz);
					NOT_ENOUGH_MEMORY(item->net);
				}
			}
		}
		fclose(fp);
	}
	globalIdCounter++;
	pthread_rwlock_unlock(&m_globalNeuro);
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

	web_restore_nets_from_file();

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
	mg_set_request_handler(ctx, DEL_URI, web_DelHandler, 0);
	mg_set_request_handler(ctx, ADD_INPUT_FILE, web_FileHandler, 0);
	mg_set_request_handler(ctx, ADD_INPUTP, web_InputHandler, 0);
	mg_set_request_handler(ctx, ADD_TRAIN_FILE, web_FileHandler, 0);
	mg_set_request_handler(ctx, ADD_TRAINP, web_InputHandler, 0);
	mg_set_request_handler(ctx, ECHO_URI, web_EchoHandler, 0);
	mg_set_request_handler(ctx, SVALL_URI, web_SaveAllHandler, 0);
	mg_set_request_handler(ctx, "/", web_IndexHandler, 0);

	memset(ports, 0, sizeof(ports));
	port_cnt = mg_get_server_ports(ctx, 32, ports);
	printf("\n%i listening ports:\n\n", port_cnt);

	for (n = 0; n < port_cnt && n < 32; n++) {
		const char *proto = ports[n].is_ssl ? "https" : "http";
		const char *host;

		if ((ports[n].protocol & 1) == 1) {
			host = "127.0.0.1";
			printf("View neuro at %s://%s:%i/\n", proto, host, ports[n].port);
			printf("Exit at %s://%s:%i%s\n", proto, host, ports[n].port,
			EXIT_URI);
			printf("\n");
		}

	}

	while (!exitNow) {
		sleep(1);
	}

	mg_stop(ctx);
	printf("Server stopped.\n");
	printf("Bye!\n");

	return EXIT_SUCCESS;
}

