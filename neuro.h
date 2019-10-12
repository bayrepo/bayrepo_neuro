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
void bayrepo_train(void *blob, int epoch);

#endif /* NEURO_H_ */
