/*
 * neuro.h
 *
 *  Created on: Oct 11, 2019
 *      Author: alexey
 */

#ifndef NEURO_H_
#define NEURO_H_

typedef enum _activation {RELU=0, SIGMOID, TANH} activation;

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
int bayrepo_write_matrix(void *blob, FILE *fp, int width, int height);

#endif /* NEURO_H_ */
