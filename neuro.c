/*
 * neuro.c
 *
 *  Created on: Oct 9, 2019
 *      Author: alexey
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>

#include "neuro.h"

typedef struct _neuro_skeleton {
	int input_l;
	int output_l;
	int hidden_l;
	int number_of_hidden;
	double alpha;
	gsl_matrix *input_m;
	gsl_matrix *output_m;
	gsl_matrix **hidden_m;
	gsl_matrix **layers;
	gsl_matrix **layers_delta;
	gsl_matrix *temp;
	gsl_matrix *train;
	int layers_numb;
	activation activ;
} neuro_skeleton;

extern unsigned long int gsl_rng_default_seed;

#define CLEAN_BLOB() bayrepo_clean_neuro((void *) blob); \
		return NULL
#define CLEAN_IF_NULL(x) if (!blob->x) { \
		CLEAN_BLOB(); \
	}

static void bayrepo_layers_clean(void *blob) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		if (data->layers) {
			int index = 0;
			for (index = 0; index < data->layers_numb; index++) {
				if (data->layers[index])
					gsl_matrix_free(data->layers[index]);
			}
			free(data->layers);
			data->layers = NULL;
		}
		if (data->layers_delta) {
			int index = 0;
			for (index = 0; index < data->layers_numb; index++) {
				if (data->layers_delta[index])
					gsl_matrix_free(data->layers_delta[index]);
			}
			data->layers_numb = 0;
			free(data->layers_delta);
			data->layers_delta = NULL;
		}
		data->layers_numb = 0;
	}
}

void bayrepo_clean_neuro(void *blob) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		if (data->input_m) {
			gsl_matrix_free(data->input_m);
			data->input_m = NULL;
		}
		if (data->output_m) {
			gsl_matrix_free(data->output_m);
			data->output_m = NULL;
		}
		if (data->train) {
			gsl_matrix_free(data->train);
			data->train = NULL;
		}
		if (data->temp) {
			gsl_matrix_free(data->temp);
			data->temp = NULL;
		}
		if (data->number_of_hidden > 0 && data->hidden_m) {
			int i;
			for (i = 0; i < data->number_of_hidden; i++) {
				if (data->hidden_m[i]) {
					gsl_matrix_free(data->hidden_m[i]);
					data->hidden_m[i] = NULL;
				}
			}
			free(data->hidden_m);
		}
		bayrepo_layers_clean(blob);
		free(data);
	}
}

static int bayrepo_random_matrix(gsl_matrix *a, int m, int n) {
	int index, jndex;
	const gsl_rng_type * T;
	gsl_rng * r;

	gsl_rng_default_seed = time(NULL);

	T = gsl_rng_knuthran2;
	r = gsl_rng_alloc(T);
	if (!r)
		return -1;

	for (index = 0; index < m; index++) {
		for (jndex = 0; jndex < n; jndex++) {
			gsl_matrix_set(a, index, jndex, gsl_rng_uniform(r));
		}
	}

	gsl_rng_free(r);
	return 0;
}

void *bayrepo_init_neuro(int input, int output, int hidden, int hidden_num,
		double alpha, activation activ) {
	int index;
	if (hidden_num < 0) {
		hidden_num = 0;
	}
	neuro_skeleton *blob = calloc(1, sizeof(neuro_skeleton));
	if (!blob)
		return NULL;
	blob->input_m = gsl_matrix_alloc(1, input);
	CLEAN_IF_NULL(input_m);
	blob->train = gsl_matrix_alloc(1, output);
	CLEAN_IF_NULL(train);

	if (hidden_num) {
		blob->output_m = gsl_matrix_alloc(hidden, output);
		CLEAN_IF_NULL(output_m);
		blob->hidden_m = (gsl_matrix **) calloc(hidden_num,
				sizeof(gsl_matrix *));
		CLEAN_IF_NULL(hidden_m);
		for (index = 1; index < hidden_num; index++) {
			blob->hidden_m[index] = gsl_matrix_alloc(hidden, hidden);
			CLEAN_IF_NULL(hidden_m[index]);
		}
		blob->hidden_m[0] = gsl_matrix_alloc(input, hidden);
		CLEAN_IF_NULL(hidden_m[0]);
		blob->temp = gsl_matrix_alloc(1, hidden);
		CLEAN_IF_NULL(temp);
	} else {
		blob->output_m = gsl_matrix_alloc(input, output);
		CLEAN_IF_NULL(output_m);
	}
	blob->alpha = alpha;
	blob->input_l = input;
	blob->output_l = output;
	blob->hidden_l = hidden;
	blob->number_of_hidden = hidden_num;

	gsl_matrix_set_zero(blob->input_m);
	gsl_matrix_set_zero(blob->train);
	if (hidden_num) {
		if (bayrepo_random_matrix(blob->output_m, hidden, output) < 0) {
			CLEAN_BLOB()
;		}
		for (index = 1; index < hidden_num; index++) {
			if (bayrepo_random_matrix(blob->hidden_m[index], hidden, hidden)<0) {
				CLEAN_BLOB();
			}
		}
		if (bayrepo_random_matrix(blob->hidden_m[0], input, hidden)<0) {
			CLEAN_BLOB();
		}
	} else {
		if (bayrepo_random_matrix(blob->output_m, input, output)<0) {
			CLEAN_BLOB();
		}
	}

	int lyr_nmb = 1 + blob->number_of_hidden ? (blob->number_of_hidden + 1) : 0;

	blob->layers = (gsl_matrix **) calloc(lyr_nmb, sizeof(gsl_matrix *));
	CLEAN_IF_NULL(layers);

	if (hidden_num) {
		for (index = 1; index < hidden_num; index++) {
			blob->layers[index] = gsl_matrix_alloc(1, hidden);
			CLEAN_IF_NULL(layers[index]);
		}
		blob->layers[0] = gsl_matrix_alloc(1, hidden);
		CLEAN_IF_NULL(layers[0]);
		blob->layers[lyr_nmb - 1] = gsl_matrix_alloc(1, output);
		CLEAN_IF_NULL(layers[lyr_nmb-1]);
	} else {
		blob->layers[0] = gsl_matrix_alloc(1, output);
		CLEAN_IF_NULL(layers[0]);
	}

	blob->layers_numb = lyr_nmb;
	blob->layers_delta = (gsl_matrix **) calloc(lyr_nmb, sizeof(gsl_matrix *));
	CLEAN_IF_NULL(layers_delta);

	if (hidden_num) {
		for (index = 1; index < hidden_num; index++) {
			blob->layers_delta[index] = gsl_matrix_alloc(1, hidden);
			CLEAN_IF_NULL(layers_delta[index]);
		}
		blob->layers_delta[0] = gsl_matrix_alloc(1, hidden);
		CLEAN_IF_NULL(layers_delta[0]);
		blob->layers_delta[lyr_nmb - 1] = gsl_matrix_alloc(1, output);
		CLEAN_IF_NULL(layers_delta[lyr_nmb-1]);
	} else {
		blob->layers_delta[0] = gsl_matrix_alloc(1, output);
		CLEAN_IF_NULL(layers_delta[0]);
	}
	blob->activ = activ;

	return (void *) blob;
}

void bayrepo_fill_input(void *blob, int position, double scaled_value) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		if (position < data->input_l) {
			gsl_matrix_set(data->input_m, 0, position, scaled_value);
		}
	}
}

void bayrepo_fill_train(void *blob, int position, double scaled_value) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		if (position < data->output_l) {
			gsl_matrix_set(data->train, 0, position, scaled_value);
		}
	}
}

static void bayrepo_zero_layers(void *blob) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		int index;
		for (index = 0; index < data->layers_numb; index++) {
			gsl_matrix_set_zero(data->layers[index]);
		}
	}
}

static double bayrepo_activation_func(double elem, neuro_skeleton *data) {
	switch (data->activ) {
	case RELU:
		return elem > 0.0 ? elem : 0.0;
	case TANH:
		return tanh(elem);
	default:
		return 1.0 / (1.0 + exp(-elem));
	}
}

static double bayrepo_activation_deriv(double elem, neuro_skeleton *data) {
	switch (data->activ) {
	case RELU:
		return elem > 0.0 ? 1.0 : 0.0;
	case TANH:
		return 1.0 - (elem * elem);
	default:
		return elem * (1.0 - elem);
	}
}

static void bayrepo_matix_customize(gsl_matrix *a, int size, neuro_skeleton *data) {
	int index;
	for (index = 0; index < size; index++) {
		gsl_matrix_set(a, 0, index,
				bayrepo_activation_func(gsl_matrix_get(a, 0, index), data));
	}
}

static void bayrepo_matix_deriv(gsl_matrix *a, int size, neuro_skeleton *data) {
	int index;
	for (index = 0; index < size; index++) {
		gsl_matrix_set(a, 0, index,
				bayrepo_activation_deriv(gsl_matrix_get(a, 0, index), data));
	}
}

void bayrepo_query(void *blob) {
	int index;
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		bayrepo_zero_layers(blob);
		if (data->hidden_l) {
			int cnt = 0;
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, data->input_m,
					data->hidden_m[0], 0.0, data->layers[0]);
			bayrepo_matix_customize(data->layers[0], data->hidden_l, data);
			cnt++;
			for (index = 1; index < data->number_of_hidden; index++) {
				gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
						data->layers[index - 1], data->hidden_m[index], 0.0,
						data->layers[index]);
				bayrepo_matix_customize(data->layers[0], data->hidden_l, data);
				cnt++;
			}
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
					data->layers[cnt - 1], data->output_m, 0.0,
					data->layers[cnt]);
			bayrepo_matix_customize(data->layers[cnt], data->output_l, data);

		} else {
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, data->input_m,
					data->output_m, 0.0, data->layers[0]);
			bayrepo_matix_customize(data->layers[0], data->output_l, data);
		}
	}
}

double bayrepo_get_result(void *blob, int position) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		if (data->layers && position < data->output_l) {
			return gsl_matrix_get(data->layers[data->layers_numb - 1], 0,
					position);
		}
	}
	return -10000.0;
}

void bayrepo_train(void *blob, int epoch) {
	int index;
	if (blob) {
		while (epoch--) {
			neuro_skeleton *data = (neuro_skeleton *) blob;
			bayrepo_query(blob);
			gsl_matrix_memcpy(data->layers_delta[data->layers_numb - 1],
					data->layers[data->layers_numb - 1]);
			gsl_matrix_sub(data->layers_delta[data->layers_numb - 1],
					data->train);
			if (data->number_of_hidden) {
				int index;
				for (index = (data->layers_numb - 2); index >= 0; index--) {
					gsl_matrix_set_zero(data->temp);
					gsl_matrix_memcpy(data->temp, data->layers[index]);
					bayrepo_matix_deriv(data->temp, data->hidden_l, data);
					if (index == (data->layers_numb - 2)) {
						gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,
								data->layers_delta[index + 1], data->output_m,
								0.0, data->layers_delta[index]);
					} else {
						gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,
								data->layers_delta[index + 1],
								data->hidden_m[index], 0.0,
								data->layers_delta[index]);
					}
					gsl_matrix_mul_elements(data->layers_delta[index],
							data->temp);
				}
				for (index = (data->layers_numb - 1); index >= 0; index--) {
					if (index == (data->layers_numb - 1)) {
						gsl_blas_dgemm(CblasTrans, CblasNoTrans, -data->alpha,
								data->layers[index - 1],
								data->layers_delta[index], 1.0, data->output_m);
					} else if (index == 0) {
						gsl_blas_dgemm(CblasTrans, CblasNoTrans, -data->alpha,
								data->input_m, data->layers_delta[index], 1.0,
								data->hidden_m[0]);
					} else {
						gsl_blas_dgemm(CblasTrans, CblasNoTrans, -data->alpha,
								data->layers[index - 1],
								data->layers_delta[index], 1.0,
								data->hidden_m[index]);
					}
				}
			} else {
				gsl_blas_dgemm(CblasTrans, CblasNoTrans, -data->alpha,
						data->input_m, data->layers_delta[0], 1.0,
						data->output_m);
			}
		}
	}
}

