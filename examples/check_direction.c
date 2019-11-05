/*
 * check_direction.c
 *
 *  Created on: Nov 3, 2019
 *      Author: alexey
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <curl/curl.h>

#include "neuro_web_client.h"

#define MAX_TRYS_DEF 1000
#define NET 13

int main(int argc, char *argv[]) {
	double inputs[4];
	double outputs[4];
	int MAX_TRYS = MAX_TRYS_DEF;
	srand(time(NULL));
	char curl_error[CURL_ERROR_SIZE];
	web_client_init("http://127.0.0.1:10111");
	int is_check = 0;
	if (argc > 1) {
		MAX_TRYS = atoi(argv[1]);
		if (MAX_TRYS <= 0)
			MAX_TRYS = MAX_TRYS_DEF;
	}
	printf("Ready for %d trys\n", MAX_TRYS);
	if (argc > 2 && !strcmp(argv[2], "check"))
		is_check = 1;
	if (argc > 2 && !strcmp(argv[2], "check_test"))
		is_check = 2;
	int index = 0;
	int errors = 0;
	int counter = 0;
	for (index = 0; index < MAX_TRYS; index++) {
		int jndex = 0;
		double max = -2.0;
		int pos = 0;
		for (jndex = 0; jndex < 4; jndex++) {
			int nmb = rand() % 10000;
			if (!nmb)
				nmb = 1;
			double number = (double) nmb / 10000.0;
			inputs[jndex] = number;
			if (number >= max) {
				max = number;
				pos = jndex;
			}
			outputs[jndex] = 0.0;
		}
		outputs[pos] = 1.0;
		if (is_check == 1) {
			printf("Try %d==============================\n", index);
			printf("Inputs: ");
			for (jndex = 0; jndex < 4; jndex++) {
				printf("[%d]=%f ", jndex, inputs[jndex]);
			}
			printf("\n");
			printf("Should get: ");
			for (jndex = 0; jndex < 4; jndex++) {
				printf("[%d]=%f ", jndex, outputs[jndex]);
			}
			printf("\n");
			web_send_inputs_to_net(NET, (double *) &inputs, 4,
					(double *) &outputs, 4, curl_error);
			printf("Got: ");
			for (jndex = 0; jndex < 4; jndex++) {
				printf("[%d]=%f ", jndex, outputs[jndex]);
			}
			printf("\n");
		} else if (is_check == 2) {
			web_send_inputs_to_net(NET, (double *) &inputs, 4,
					(double *) &outputs, 4, curl_error);
			int new_pos = 0;
			max = -2;
			for (jndex = 0; jndex < 4; jndex++) {
				if (outputs[jndex] >= max) {
					max = outputs[jndex];
					new_pos = jndex;
				}
			}
			if (pos != new_pos) {
				errors++;
			}
			if (index % (MAX_TRYS / 20) == 0)
				printf("Check inputs %d\n", index);

		} else {
			web_send_train_to_net(NET, (double *) &inputs, 4, (double *) &outputs,
					4, curl_error);
			if (index % (MAX_TRYS / 20) == 0)
				printf("Train inputs %d\n", index);
		}
		counter++;
	}
	if (is_check == 2) {
		printf("Check %d results: errors = %d\n", counter, errors);
	}
	web_client_clean();
	return 0;
}
