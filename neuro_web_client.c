/*
 * neuro_web_client.c
 *
 *  Created on: Oct 27, 2019
 *      Author: alexey
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <string.h>
#include "neuro_urls.h"
#include "neuro_web_client.h"

#define MAX_ADDR_LEN 4096
#define MAX_FIELD_NAME 256
#define WEB_INITIAL_BUFFER_SIZE (1)
#define WEB_MAXIMUM_BUFFER_SIZE (8 * 1024 * 1024)

static char web_addr[MAX_ADDR_LEN];

struct web_buffer {
	unsigned char *memory;
	size_t size;
	size_t used;
};

static size_t web_curl_write(char *ptr, size_t size, size_t nmemb,
		void *userdata) {
	struct web_buffer *buffer = userdata;
	size_t needed = size * nmemb;

	if (needed > (buffer->size - buffer->used)) {
		unsigned char *new_memory;
		size_t new_size = 2 * buffer->size;
		while (needed > (new_size - buffer->used)) {
			new_size *= 2;
			if (new_size > (WEB_MAXIMUM_BUFFER_SIZE)) {
				return 0;
			}
		}
		new_memory = realloc(buffer->memory, new_size);
		if (!new_memory) {
			return 0;
		}
		buffer->memory = new_memory;
		buffer->size = new_size;
	}
	memcpy(buffer->memory + buffer->used, ptr, needed);
	buffer->used += needed;
	return needed;
}

void web_client_init(char *addr) {
	curl_global_init(CURL_GLOBAL_ALL);
	strncpy(web_addr, addr ? addr : "http://127.0.0.1:10111", MAX_ADDR_LEN);
}

void web_client_clean() {
	curl_global_cleanup();
}

int web_send_inputs_to_net(int net_id, double *inputs, int len_inputs,
		double *outputs, int len_outputs, char *error) {
	CURL *curl;
	CURLcode res;
	char uri[MAX_ADDR_LEN];
	char name[MAX_ADDR_LEN];
	char value[MAX_ADDR_LEN];
	int index;
	struct curl_httppost *formpost = NULL;
	struct curl_httppost *lastptr = NULL;
	struct curl_slist *headerlist = NULL;
	static const char buf[] = "";
	for (index = 0; index < len_outputs; index++) {
		outputs[index] = 0.0;
	}
	curl = curl_easy_init();
	if (curl) {
		for (index = 0; index < len_inputs; index++) {
			snprintf(name, MAX_FIELD_NAME, "fin2_%d", index);
			snprintf(value, MAX_FIELD_NAME, "%f", inputs[index]);
			curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, name,
					CURLFORM_COPYCONTENTS, value, CURLFORM_END);
		}
		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "ajax",
				CURLFORM_COPYCONTENTS, "y", CURLFORM_END);
		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "submit",
				CURLFORM_COPYCONTENTS, "Send", CURLFORM_END);
		headerlist = curl_slist_append(headerlist, buf);
		headerlist = curl_slist_append(headerlist, "cache-control: no-cache");

		snprintf(uri, MAX_ADDR_LEN, "%s%s/%d", web_addr, ADD_INPUTP, net_id);
		curl_easy_setopt(curl, CURLOPT_URL, uri);
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
		curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");

		struct web_buffer buffer;
		buffer.memory = malloc(WEB_INITIAL_BUFFER_SIZE);
		buffer.size = WEB_INITIAL_BUFFER_SIZE;
		buffer.used = 0;
		if (!buffer.memory) {
			curl_easy_cleanup(curl);
			curl_formfree(formpost);
			curl_slist_free_all(headerlist);
			strcpy(error, "curl: not enough memory");
			return -1;
		}

		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, web_curl_write);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

		int ret_code = 0;
		res = curl_easy_perform(curl);
		if (res != CURLE_OK && res != CURLE_PARTIAL_FILE) {
			strcpy(error, curl_easy_strerror(res));
			ret_code = -1;
		}
		curl_easy_cleanup(curl);
		curl_formfree(formpost);
		curl_slist_free_all(headerlist);

		if (ret_code == 0) {
			FILE *fp = fmemopen((void*) buffer.memory, buffer.used, "r");
			if (fp) {
				char result_buf[MAX_ADDR_LEN];
				int is_result = 0;
				while (!feof(fp)) {
					if (fgets(result_buf, MAX_ADDR_LEN, fp)) {
						if (strstr(result_buf, "TYPE:RESULT")) {
							is_result = 1;
						} else if (is_result && strchr(result_buf, ':')) {
							char *ptr = result_buf;
							char *ptr2 = strchr(result_buf, ':');
							char *tptr;
							*ptr2 = 0;
							ptr2++;
							int nmb = atoi(ptr);
							double val = strtod(ptr2, &tptr);
							if (nmb < len_outputs) {
								outputs[nmb] = val;
							}
						}
					}
				}
				if (!is_result) {
					ret_code = -1;
					strcpy(error, "curl: incorrect result");
				}
			} else {
				ret_code = -1;
				strcpy(error, "curl: result parsing error");
			}
		}

		if (buffer.memory)
			free(buffer.memory);
		return ret_code;
	}
	strcpy(error, "curl: init error");
	return -1;
}

int web_send_train_to_net(int net_id, double *inputs, int len_inputs,
		double *outputs, int len_outputs, char *error) {
	CURL *curl;
	CURLcode res;
	char uri[MAX_ADDR_LEN];
	char name[MAX_ADDR_LEN];
	char value[MAX_ADDR_LEN];
	int index;
	struct curl_httppost *formpost = NULL;
	struct curl_httppost *lastptr = NULL;
	struct curl_slist *headerlist = NULL;
	static const char buf[] = "";
	curl = curl_easy_init();
	if (curl) {
		for (index = 0; index < len_inputs; index++) {
			snprintf(name, MAX_FIELD_NAME, "tin2_%d", index);
			snprintf(value, MAX_FIELD_NAME, "%f", inputs[index]);
			curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, name,
					CURLFORM_COPYCONTENTS, value, CURLFORM_END);
		}
		for (index = 0; index < len_outputs; index++) {
			snprintf(name, MAX_FIELD_NAME, "tine2_%d", index);
			snprintf(value, MAX_FIELD_NAME, "%f", outputs[index]);
			curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, name,
					CURLFORM_COPYCONTENTS, value, CURLFORM_END);
		}
		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "ajax",
				CURLFORM_COPYCONTENTS, "y", CURLFORM_END);
		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "submit",
				CURLFORM_COPYCONTENTS, "Send", CURLFORM_END);
		headerlist = curl_slist_append(headerlist, buf);
		headerlist = curl_slist_append(headerlist, "cache-control: no-cache");

		snprintf(uri, MAX_ADDR_LEN, "%s%s/%d", web_addr, ADD_TRAINP, net_id);
		curl_easy_setopt(curl, CURLOPT_URL, uri);
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
		curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");

		struct web_buffer buffer;
		buffer.memory = malloc(WEB_INITIAL_BUFFER_SIZE);
		buffer.size = WEB_INITIAL_BUFFER_SIZE;
		buffer.used = 0;
		if (!buffer.memory) {
			curl_easy_cleanup(curl);
			curl_formfree(formpost);
			curl_slist_free_all(headerlist);
			strcpy(error, "curl: not enough memory");
			return -1;
		}

		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, web_curl_write);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

		int ret_code = 0;
		res = curl_easy_perform(curl);
		if (res != CURLE_OK && res != CURLE_PARTIAL_FILE) {
			strcpy(error, curl_easy_strerror(res));
			ret_code = -1;
		}
		curl_easy_cleanup(curl);
		curl_formfree(formpost);
		curl_slist_free_all(headerlist);

		if (ret_code == 0) {
			FILE *fp = fmemopen((void*) buffer.memory, buffer.used, "r");
			if (fp) {
				char result_buf[MAX_ADDR_LEN];
				int is_result = 0;
				while (!feof(fp)) {
					if (fgets(result_buf, MAX_ADDR_LEN, fp)) {
						if (strstr(result_buf, "TYPE:TRAINING")) {
							is_result = 1;
						}
					}
				}
				if (!is_result) {
					ret_code = -1;
					strcpy(error, "curl: incorrect result");
				}
			} else {
				ret_code = -1;
				strcpy(error, "curl: result parsing error");
			}
		}

		if (buffer.memory)
			free(buffer.memory);
		return ret_code;
	}
	strcpy(error, "curl: init error");
	return -1;
}

#ifdef STANDALONE
int main() {
	double inputs[2];
	double outputs[1];
	char curl_error[CURL_ERROR_SIZE];
	inputs[0] = 0.0;
	inputs[1] = 1.0;
	web_client_init("http://127.0.0.1:10111");
	web_send_inputs_to_net(1, (double *) &inputs, 2, (double *) &outputs, 1,
			curl_error);
	printf("Result %f\n", outputs[0]);
	inputs[0] = 1.0;
	inputs[1] = 1.0;
	outputs[0] = 1.0;
	web_send_train_to_net(2, (double *) &inputs, 2, (double *) &outputs, 1,
			curl_error);
	web_client_clean();
	return 0;
}
#endif
