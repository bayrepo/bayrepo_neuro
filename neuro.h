/*
 * neuro.h
 *
 *  Created on: Oct 11, 2019
 *      Author: alexey
 */

#ifndef NEURO_H_
#define NEURO_H_

typedef void (*bayrepo_printer_t)(void *);
typedef void (*bayrepo_printerfunc_t)(void *, double);
typedef void (*bayrepo_printerhdr_t)(void *, char *, int);

typedef struct __bayrepo_decorator {
	void *user_data;
	bayrepo_printerhdr_t table_header;
	bayrepo_printer_t table_footer;
	bayrepo_printer_t pre_column;
	bayrepo_printer_t post_column;
	bayrepo_printerfunc_t print_func;
} bayrepo_decorator;

typedef struct __bayrepo_mem_encode {
	char *buffer;
	size_t size;
} bayrepo_mem_encode;

typedef enum _activation {RELU=0, SIGMOID, TANH, NOACTIV, DEFLT} activation;

void bayrepo_clean_neuro(void *blob);
void *bayrepo_init_neuro(int input, int output, int hidden, int hidden_num,
		double alpha, activation activ);
void bayrepo_fill_input(void *blob, int position, double scaled_value);
void bayrepo_fill_train(void *blob, int position, double scaled_value);
void bayrepo_query(void *blob);
double bayrepo_get_result(void *blob, int position);
void bayrepo_train(void *blob, int epoch, int use_dropout);
void bayrepo_fill_hidden(void *blob, int index, int position_x, int position_y,
		double scaled_value);
void bayrepo_fill_outm(void *blob, int position_x, int position_y,
		double scaled_value);
int bayrepo_write_matrix(void *blob, FILE *fp, int width, int height, bayrepo_mem_encode *buffer);
void bayrepo_set_layer_activ(void * blob, int layer_number, activation activ);
activation bayrepo_get_layer_func(void *blob, int lyn);
void bayrepo_print_matrix_custom(void *blob, bayrepo_decorator *decor);

#endif /* NEURO_H_ */
