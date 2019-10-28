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
#include <png.h>
#include <malloc.h>

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
	gsl_matrix **dropuot;
	gsl_matrix **layers;
	gsl_matrix **layers_delta;
	gsl_matrix *temp;
	gsl_matrix *train;
	gsl_matrix *error;
	int layers_numb;
	activation activ;
	activation *activ_l;
} neuro_skeleton;

const gsl_rng_type * T;
gsl_rng * r;

extern unsigned long int gsl_rng_default_seed;

#define CLEAN_BLOB() bayrepo_clean_neuro((void *) blob); \
		return NULL
#define CLEAN_IF_NULL(x) if (!blob->x) { \
		CLEAN_BLOB(); \
	}

#define ISDEBUGINFO_BEG() if (getenv("MATRIXD") && !strcmp(getenv("MATRIXD"),"1")) {
#define ISDEBUGINFO_END() }
#define DEBUGINFO(A, B, C) ISDEBUGINFO_BEG()\
		bayrepo_print_matrix(A, B, C); \
		ISDEBUGINFO_END()

#define MAX_LAYER_NAME_LEN 1024
#define MAX_BUFFER_LEN 4096

static void bayrepo_print_matrix(gsl_matrix *a, const char *matrix_name,
		int index_matrix) {
	int index, jndex;
	if (index_matrix >= 0) {
		printf("Matrix %s[%d]:\n", matrix_name, index_matrix);
	} else {
		printf("Matrix %s:\n", matrix_name);
	}
	for (index = 0; index < a->size1; index++) {
		printf("==");
		for (jndex = 0; jndex < a->size2; jndex++) {
			printf("%.3f   ", gsl_matrix_get(a, index, jndex));
		}
		printf("==\n");
	}
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
		if (data->error) {
			gsl_matrix_free(data->error);
			data->error = NULL;
		}
		if (data->activ_l) {
			free(data->activ_l);
			data->activ_l = NULL;
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
		if (data->number_of_hidden > 0 && data->dropuot) {
			int i;
			for (i = 0; i < data->number_of_hidden; i++) {
				if (data->dropuot[i]) {
					gsl_matrix_free(data->dropuot[i]);
					data->dropuot[i] = NULL;
				}
			}
			free(data->dropuot);
		}
		bayrepo_layers_clean(blob);
		free(data);
	}
	if (r) {
		gsl_rng_free(r);
		r = NULL;
	}
}

void bayrepo_set_layer_activ(void * blob, int layer_number, activation activ) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		if (data->number_of_hidden) {
			if (layer_number <= data->number_of_hidden) {
				data->activ_l[layer_number] = activ;
			}
		} else {
			if (!layer_number) {
				data->activ_l[0] = activ;
			}
		}
	}
}

static int bayrepo_random_matrix(gsl_matrix *a, int m, int n, int is_negative) {
	int index, jndex;

	if (!r)
		return -1;

	for (index = 0; index < m; index++) {
		for (jndex = 0; jndex < n; jndex++) {
			gsl_matrix_set(a, index, jndex,
					gsl_rng_uniform(r) - (is_negative ? 0.5 : 0.0));
		}
	}

	return 0;
}

void *bayrepo_init_neuro(int input, int output, int hidden, int hidden_num,
		double alpha, activation activ) {
	int index;

	gsl_rng_default_seed = time(NULL);

	T = gsl_rng_knuthran2;
	r = gsl_rng_alloc(T);

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
	blob->error = gsl_matrix_alloc(1, output);
	CLEAN_IF_NULL(error);

	if (hidden_num) {
		blob->output_m = gsl_matrix_alloc(hidden, output);
		CLEAN_IF_NULL(output_m);
		blob->hidden_m = (gsl_matrix **) calloc(hidden_num,
				sizeof(gsl_matrix *));
		CLEAN_IF_NULL(hidden_m);
		blob->dropuot = (gsl_matrix **) calloc(hidden_num,
				sizeof(gsl_matrix *));
		CLEAN_IF_NULL(dropuot);
		for (index = 1; index < hidden_num; index++) {
			blob->hidden_m[index] = gsl_matrix_alloc(hidden, hidden);
			CLEAN_IF_NULL(hidden_m[index]);
			blob->dropuot[index] = gsl_matrix_alloc(1, hidden);
			CLEAN_IF_NULL(dropuot[index]);
		}
		blob->hidden_m[0] = gsl_matrix_alloc(input, hidden);
		CLEAN_IF_NULL(hidden_m[0]);
		blob->dropuot[0] = gsl_matrix_alloc(1, hidden);
		CLEAN_IF_NULL(dropuot[0]);
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
		if (bayrepo_random_matrix(blob->output_m, hidden, output,
				(activ == RELU ? 0 : 1)) < 0) {
			CLEAN_BLOB()
;		}
		for (index = 1; index < hidden_num; index++) {
			if (bayrepo_random_matrix(blob->hidden_m[index], hidden, hidden, (activ==RELU?0:1))<0) {
				CLEAN_BLOB();
			}
		}
		if (bayrepo_random_matrix(blob->hidden_m[0], input, hidden, (activ==RELU?0:1))<0) {
			CLEAN_BLOB();
		}
	} else {
		if (bayrepo_random_matrix(blob->output_m, input, output, (activ==RELU?0:1))<0) {
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
	blob->activ_l = calloc(blob->number_of_hidden + 1, sizeof(activation));
	CLEAN_IF_NULL(activ_l);
	for (index = 0; index < (blob->number_of_hidden + 1); index++) {
		if (index == blob->number_of_hidden) {
			blob->activ_l[index] = NOACTIV;
		} else {
			blob->activ_l[index] = DEFLT;
		}
	}

	return (void *) blob;
}

static void bayrepo_fill_dropout(gsl_matrix *a, int m, int n) {
	int index, jndex;
	if (!r) {
		for (index = 0; index < m; index++) {
			for (jndex = 0; jndex < n; jndex++) {
				gsl_matrix_set(a, index, jndex, 1.0);
			}
		}
		return;
	}

	for (index = 0; index < m; index++) {
		for (jndex = 0; jndex < n; jndex++) {
			gsl_matrix_set(a, index, jndex, gsl_rng_uniform_int(r, 2) * 1.0);
		}
	}

	return;
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

void bayrepo_fill_hidden(void *blob, int index, int position_x, int position_y,
		double scaled_value) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		gsl_matrix_set(data->hidden_m[index], position_x, position_y,
				scaled_value);
	}
}

void bayrepo_fill_outm(void *blob, int position_x, int position_y,
		double scaled_value) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		gsl_matrix_set(data->output_m, position_x, position_y, scaled_value);
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

static double bayrepo_activation_func(double elem, neuro_skeleton *data,
		int lyn) {
	activation act = data->activ;
	if ((lyn < (data->number_of_hidden + 1)) && (data->activ_l[lyn] != DEFLT)) {
		act = data->activ_l[lyn];
	}
	switch (act) {
	case RELU:
		return elem > 0.0 ? elem : 0.0;
	case TANH:
		return tanh(elem);
	case SIGMOID:
		return 1.0 / (1.0 + exp(-elem));
	default:
		return elem;
	}
}

activation bayrepo_get_layer_func(void *blob, int lyn) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		if ((lyn < (data->number_of_hidden + 1))
				&& (data->activ_l[lyn] != DEFLT)) {
			return data->activ_l[lyn];
		}
	}
	return NOACTIV;
}

activation bayrepo_get_sublayer_func(void *blob, int lyn) {
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		if (lyn < (data->number_of_hidden + 1)) {
			return data->activ_l[lyn];
		}
	}
	return NOACTIV;
}

static double bayrepo_activation_deriv(double elem, neuro_skeleton *data,
		int lyn) {
	activation act = data->activ;
	if ((lyn < (data->number_of_hidden + 1)) && (data->activ_l[lyn] != DEFLT)) {
		act = data->activ_l[lyn];
	}
	switch (act) {
	case RELU:
		return elem > 0.0 ? 1.0 : 0.0;
	case TANH:
		return 1.0 - (elem * elem);
	case SIGMOID:
		return elem * (1.0 - elem);
	default:
		return elem;
	}
}

static void bayrepo_matix_customize(gsl_matrix *a, int size,
		neuro_skeleton *data, int lyn) {
	int index;
	for (index = 0; index < size; index++) {
		gsl_matrix_set(a, 0, index,
				bayrepo_activation_func(gsl_matrix_get(a, 0, index), data,
						lyn));
	}
}

static void bayrepo_matix_deriv(gsl_matrix *a, int size, neuro_skeleton *data,
		int lyn) {
	int index;
	for (index = 0; index < size; index++) {
		gsl_matrix_set(a, 0, index,
				bayrepo_activation_deriv(gsl_matrix_get(a, 0, index), data,
						lyn));
	}
}

static void bayrepo_query_internal(void *blob, int dropout) {
	int index;
	if (blob) {
		ISDEBUGINFO_BEG()
			printf("=====================Query=========================\n");
		ISDEBUGINFO_END()
		neuro_skeleton *data = (neuro_skeleton *) blob;
		bayrepo_zero_layers(blob);

		if (data->number_of_hidden) {
			if (dropout == 1) {
				for (index = 0; index < data->number_of_hidden; index++) {
					bayrepo_fill_dropout(data->dropuot[index], 1,
							data->hidden_l);
				}
			}
			int cnt = 0;
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, data->input_m,
					data->hidden_m[0], 0.0, data->layers[0]);
			bayrepo_matix_customize(data->layers[0], data->hidden_l, data, 0);
			if (dropout == 1) {
				gsl_matrix_mul_elements(data->layers[0], data->dropuot[0]);
				gsl_matrix_scale(data->layers[0], 2.0);
			}
			cnt++;
			for (index = 1; index < data->number_of_hidden; index++) {
				gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
						data->layers[index - 1], data->hidden_m[index], 0.0,
						data->layers[index]);
				bayrepo_matix_customize(data->layers[index], data->hidden_l,
						data, index);
				if (dropout == 1) {
					gsl_matrix_mul_elements(data->layers[index],
							data->dropuot[index]);
					gsl_matrix_scale(data->layers[index], 2.0);
				}
				cnt++;
			}
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
					data->layers[cnt - 1], data->output_m, 0.0,
					data->layers[cnt]);
			bayrepo_matix_customize(data->layers[cnt], data->output_l, data,
					cnt);

		} else {
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, data->input_m,
					data->output_m, 0.0, data->layers[0]);
			bayrepo_matix_customize(data->layers[0], data->output_l, data, 0);
		}

		DEBUGINFO(data->input_m, "INPUTM", -1);
		if (data->number_of_hidden) {
			for (index = 0; index < data->number_of_hidden; index++) {
				DEBUGINFO(data->hidden_m[index], "HIDDEN", index);
			}
		}
		DEBUGINFO(data->output_m, "OUTPUTM", -1);
		for (index = 0; index < data->layers_numb; index++) {
			DEBUGINFO(data->layers[index], "LAYER", index);
		}
	}
}

void bayrepo_query(void *blob) {
	bayrepo_query_internal(blob, 0);
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

double bayrepo_get_sum(gsl_matrix *a) {
	double result = 0.0;
	int index, jndex = 0;
	for (index = 0; index < a->size1; index++) {
		for (jndex = 0; jndex < a->size2; jndex++) {
			result += gsl_matrix_get(a, index, jndex);
		}
	}
	return result;
}

void bayrepo_train(void *blob, int epoch, int use_dropout) {
	int index;
	if (blob) {
		while (epoch--) {
			ISDEBUGINFO_BEG()
				printf(
						"=====================Epoch %d=========================\n",
						epoch);
			ISDEBUGINFO_END()
			neuro_skeleton *data = (neuro_skeleton *) blob;
			bayrepo_query_internal(blob, use_dropout);
			gsl_matrix_memcpy(data->layers_delta[data->layers_numb - 1],
					data->layers[data->layers_numb - 1]);
			ISDEBUGINFO_BEG()
				gsl_matrix_memcpy(data->error,
						data->layers[data->layers_numb - 1]);
			ISDEBUGINFO_END();
			gsl_matrix_sub(data->layers_delta[data->layers_numb - 1],
					data->train);
			ISDEBUGINFO_BEG()
				gsl_matrix_sub(data->error, data->train);
				gsl_matrix_mul_elements(data->error, data->error);
				printf("=======>Error=%f\n", bayrepo_get_sum(data->error));
			ISDEBUGINFO_END();
			if (data->number_of_hidden) {
				int index;
				for (index = (data->layers_numb - 2); index >= 0; index--) {
					gsl_matrix_set_zero(data->temp);
					gsl_matrix_memcpy(data->temp, data->layers[index]);
					bayrepo_matix_deriv(data->temp, data->hidden_l, data,
							index);
					if (index == (data->layers_numb - 2)) {
						gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,
								data->layers_delta[index + 1], data->output_m,
								0.0, data->layers_delta[index]);
					} else {
						gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,
								data->layers_delta[index + 1],
								data->hidden_m[index + 1], 0.0,
								data->layers_delta[index]);
					}
					gsl_matrix_mul_elements(data->layers_delta[index],
							data->temp);
				}

				if (use_dropout) {
					for (index = 0; index < data->number_of_hidden; index++) {
						gsl_matrix_mul_elements(data->layers_delta[index],
								data->dropuot[index]);
					}
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
			for (index = 0; index < data->layers_numb; index++) {
				DEBUGINFO(data->layers_delta[index], "LAYER_DELTA", index);
			}
		}
	}
}

static void byrepo_png_write_data(png_structp png_ptr, png_bytep data,
		png_size_t length) {

	bayrepo_mem_encode* p = (bayrepo_mem_encode*) png_get_io_ptr(png_ptr);
	size_t nsize = p->size + length;

	if (p->buffer)
		p->buffer = realloc(p->buffer, nsize);
	else
		p->buffer = malloc(nsize);

	if (!p->buffer)
		png_error(png_ptr, "Write Error");

	memcpy(p->buffer + p->size, data, length);
	p->size += length;
}

static void bayrepo_png_flush(png_structp png_ptr) {
}

int bayrepo_write_matrix(void *blob, FILE *fp, int width, int height,
		bayrepo_mem_encode *buffer) {
	int code = 0;
	if (buffer) {
		buffer->buffer = NULL;
		buffer->size = 0;
	}
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep row = NULL;
	if ((blob == NULL) || (fp == NULL)) {
		return -1;
	}

	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		code = -3;
		goto finalise;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		code = -4;
		goto finalise;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		code = -5;
		goto finalise;
	}

	if (buffer) {
		png_set_write_fn(png_ptr, buffer, byrepo_png_write_data,
				bayrepo_png_flush);
	} else {
		png_init_io(png_ptr, fp);
	}

	neuro_skeleton *data = (neuro_skeleton *) blob;

	int picSize = (height + 3) + (height + 3) * data->number_of_hidden;

	png_set_IHDR(png_ptr, info_ptr, width, picSize, 8, PNG_COLOR_TYPE_GRAY,
	PNG_INTERLACE_NONE,
	PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	row = (png_bytep) malloc(3 * width * sizeof(png_byte));
	if (!row) {
		code = -5;
		goto finalise;
	}

	int index = 0;
	for (index = 0; index < data->number_of_hidden + 1; index++) {
		gsl_matrix * m = NULL;
		if (data->number_of_hidden) {
			m = (index == data->number_of_hidden) ?
					data->output_m : data->hidden_m[index];
		} else {
			m = data->output_m;
		}

		double rangeMax = gsl_matrix_max(m);
		double rangeMin = gsl_matrix_min(m);

		if (((width / m->size1) == 0) || ((height / m->size2) == 0)) {
			return -2;
		}

		int sizeX = width / m->size1;
		int sizeY = height / m->size2;

		int x, y;
		for (y = 0; y < height; y++) {
			for (x = 0; x < width; x++) {
				int curX = x / sizeX;
				int curY = y / sizeY;
				if ((curX >= m->size1) || (curY >= m->size2)) {
					row[x] = (png_byte) 255;
				} else {
					png_byte res_color = (png_byte) ((gsl_matrix_get(m, curX,
							curY) - rangeMin) / (rangeMax - rangeMin) * 255.0);
					if (res_color > 255.0)
						res_color = 255.0;
					row[x] = (png_byte) (255.0 - res_color);
				}
			}
			png_write_row(png_ptr, row);
		}
		for (y = 0; y < 3; y++) {
			for (x = 0; x < width; x++) {
				if (y != 1) {
					row[x] = (png_byte) 255;
				} else {
					row[x] = (png_byte) 0;
				}
			}
			png_write_row(png_ptr, row);
		}
	}

	png_write_end(png_ptr, NULL);

	finalise: if (info_ptr != NULL)
		png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (png_ptr != NULL)
		png_destroy_write_struct(&png_ptr, (png_infopp) NULL);
	if (row != NULL)
		free(row);
	return code;

}

void bayrepo_print_matrix_custom(void *blob, bayrepo_decorator *decor) {
	char layer_name[MAX_LAYER_NAME_LEN];
	if (blob) {
		neuro_skeleton *data = (neuro_skeleton *) blob;
		int index = 0;
		for (index = 0; index < (data->number_of_hidden + 1); index++) {
			gsl_matrix * m = NULL;
			if (data->number_of_hidden) {
				m = (index == data->number_of_hidden) ?
						data->output_m : data->hidden_m[index];
				if (index == data->number_of_hidden) {
					snprintf(layer_name, MAX_LAYER_NAME_LEN, "Output layer");
				} else {
					snprintf(layer_name, MAX_LAYER_NAME_LEN, "Hidden layer %d",
							index);
				}
			} else {
				m = data->output_m;
				snprintf(layer_name, MAX_LAYER_NAME_LEN, "Output layer");
			}
			int posX, posY;
			if (decor->table_header)
				decor->table_header(decor->user_data, layer_name, m->size1);
			for (posX = 0; posX < m->size2; posX++) {
				if (decor->pre_column)
					decor->pre_column(decor->user_data);
				for (posY = 0; posY < m->size1; posY++) {
					if (decor->print_func)
						decor->print_func(decor->user_data,
								gsl_matrix_get(m, posY, posX));
				}
				if (decor->post_column)
					decor->post_column(decor->user_data);
			}
			if (decor->table_footer)
				decor->table_footer(decor->user_data);
		}
	}
}

int input_l;
int output_l;
int hidden_l;
int number_of_hidden;
double alpha;
gsl_matrix *input_m;
gsl_matrix *output_m;
gsl_matrix **hidden_m;
gsl_matrix **dropuot;
gsl_matrix **layers;
gsl_matrix **layers_delta;
gsl_matrix *temp;
gsl_matrix *train;
gsl_matrix *error;
int layers_numb;
activation activ;
activation *activ_l;

/*
 * Save format
 *
 *  [int input_l]\n
 *  [int output_l]\n
 *  [int hidden_l]\n
 *  [int number_of_hidden]\n
 *  [double alpha]\n
 *  [int activ]\n
 *  [ACTBEG]\n
 *  [int layer_number]:[int activ]\n
 *  [ACTEND]\n
 *  [HIDDEN_BEG]\n
 *  [int X]:[int Y]:[double value]\n
 *  [HIDDEN_END]
 *  [OUTPUT_BEG]\n
 *  [int X]:[int Y]:[double value]\n
 *  [OUTPUT_END]\n
 */

static char *bayrepo_reallocate_buffer(char *buffer, char *newdata, int *len,
		int newlen) {
	char *new_buff = (char *) malloc(*len + newlen);
	if (!new_buff) {
		return NULL;
	}
	if (buffer) {
		memcpy(new_buff, buffer, *len);
	}
	memcpy(new_buff + *len, newdata, newlen);
	if (buffer)
		free(buffer);
	*len = *len + newlen;
	return new_buff;
}

int bayrepo_save_to_buffer(void *blob, char **buffer) {
	char buff[MAX_LAYER_NAME_LEN];
	int index;
	if (blob) {
		int len = 0;
		char *ptr = NULL;
		neuro_skeleton *data = (neuro_skeleton *) blob;
		snprintf(buff, MAX_LAYER_NAME_LEN, "%d\n", data->input_l);
		ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len, strlen(buff));
		snprintf(buff, MAX_LAYER_NAME_LEN, "%d\n", data->output_l);
		ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len, strlen(buff));
		snprintf(buff, MAX_LAYER_NAME_LEN, "%d\n", data->hidden_l);
		ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len, strlen(buff));
		snprintf(buff, MAX_LAYER_NAME_LEN, "%d\n", data->number_of_hidden);
		ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len, strlen(buff));
		snprintf(buff, MAX_LAYER_NAME_LEN, "%f\n", data->alpha);
		ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len, strlen(buff));
		snprintf(buff, MAX_LAYER_NAME_LEN, "%d\n", (int) data->activ);
		ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len, strlen(buff));
		ptr = bayrepo_reallocate_buffer(ptr, (char *) "[ACTBEG]\n", &len,
				strlen("[ACTBEG]\n"));
		for (index = 0; index < (data->number_of_hidden + 1); index++) {
			snprintf(buff, MAX_LAYER_NAME_LEN, "%d:%d\n", index,
					(int) data->activ_l[index]);
			ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len,
					strlen(buff));
		}
		ptr = bayrepo_reallocate_buffer(ptr, (char *) "[ACTEND]\n", &len,
				strlen("[ACTEND]\n"));
		for (index = 0; index < data->number_of_hidden; index++) {
			ptr = bayrepo_reallocate_buffer(ptr, (char *) "[HIDDEN_BEG]\n",
					&len, strlen("[HIDDEN_BEG]\n"));
			int x, y;
			for (x = 0; x < data->hidden_m[index]->size1; x++) {
				for (y = 0; y < data->hidden_m[index]->size2; y++) {
					snprintf(buff, MAX_LAYER_NAME_LEN, "%d:%d:%f\n", x, y,
							gsl_matrix_get(data->hidden_m[index], x, y));
					ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len,
							strlen(buff));
				}
			}
			ptr = bayrepo_reallocate_buffer(ptr, (char *) "[HIDDEN_END]\n",
					&len, strlen("[HIDDEN_END]\n"));
		}
		ptr = bayrepo_reallocate_buffer(ptr, (char *) "[OUTPUT_BEG]\n", &len,
				strlen("[OUTPUT_BEG]\n"));
		int x, y;
		for (x = 0; x < data->output_m->size1; x++) {
			for (y = 0; y < data->output_m->size2; y++) {
				snprintf(buff, MAX_LAYER_NAME_LEN, "%d:%d:%f\n", x, y,
						gsl_matrix_get(data->output_m, x, y));
				ptr = bayrepo_reallocate_buffer(ptr, (char *) buff, &len,
						strlen(buff));
			}
		}
		ptr = bayrepo_reallocate_buffer(ptr, (char *) "[OUTPUT_END]\n", &len,
				strlen("[OUTPUT_END]\n"));
		*buffer = ptr;
		return len;
	}
	return 0;
}

void *bayrepo_restore_buffer(char *buffer, int buffer_len) {
	FILE *fp = fmemopen((void*) buffer, buffer_len, "r");
	int input_l = 0;
	int output_l = 0;
	int hidden_l = 0;
	int numb_of_hidden = 0;
	activation act = RELU;
	int act_beg = 0;
	double alpha = 0.0;
	void *blob = NULL;
	int is_hidden = 0;
	int hidden_index = 0;
	int is_out = 0;
	if (fp) {
		char result_buf[MAX_BUFFER_LEN];
		int counter = 0;

		while (!feof(fp)) {
			if (fgets(result_buf, MAX_BUFFER_LEN, fp)) {
				switch (counter) {
				case 0: {
					result_buf[strlen(result_buf) - 1] = 0;
					input_l = atoi(result_buf);
				}
					break;
				case 1: {
					result_buf[strlen(result_buf) - 1] = 0;
					output_l = atoi(result_buf);
				}
					break;
				case 2: {
					result_buf[strlen(result_buf) - 1] = 0;
					hidden_l = atoi(result_buf);
				}
					break;
				case 3: {
					result_buf[strlen(result_buf) - 1] = 0;
					numb_of_hidden = atoi(result_buf);
				}
					break;
				case 4: {
					char *tptr;
					result_buf[strlen(result_buf) - 1] = 0;
					alpha = strtod(result_buf, &tptr);
				}
					break;
				case 5: {
					result_buf[strlen(result_buf) - 1] = 0;
					act = (activation) atoi(result_buf);
					if (input_l > 0 && output_l > 0 && hidden_l >= 0
							&& numb_of_hidden >= 0 && alpha > 0.0) {
						blob = bayrepo_init_neuro(input_l, output_l, hidden_l,
								numb_of_hidden, alpha, act);
					}
				}
					break;
				default: {
					if (!blob)
						return NULL;
					if (strstr(result_buf, "[ACTBEG]")) {
						act_beg = 1;
					} else if (strstr(result_buf, "[ACTEND]")) {
						act_beg = 0;
					} else if (act_beg) {
						result_buf[strlen(result_buf) - 1] = 0;
						char *ptr = strchr(result_buf, ':');
						if (ptr) {
							*ptr = 0;
							ptr++;
							int ly = atoi(result_buf);
							int ac = atoi(ptr);
							bayrepo_set_layer_activ(blob, ly, (activation) ac);
						}
					} else if (strstr(result_buf, "[HIDDEN BEG]")) {
						is_hidden = 1;
					} else if (strstr(result_buf, "[HIDDEN END]")) {
						hidden_index++;
						is_hidden = 0;
					} else if (is_hidden) {
						result_buf[strlen(result_buf) - 1] = 0;
						char *ptr1 = strchr(result_buf, ':');
						if (ptr1) {
							*ptr1 = 0;
							ptr1++;
							char *ptr2 = strchr(ptr1, ':');
							if (ptr2) {
								*ptr2 = 0;
								ptr2++;
								char *tptr = NULL;
								int x = atoi(result_buf);
								int y = atoi(ptr1);
								double vl = strtod(ptr2, &tptr);
								bayrepo_fill_hidden(blob, hidden_index, x, y,
										vl);
							}
						}
					} else if (strstr(result_buf, "[OUTPUT_BEG]")) {
						is_out = 1;
					} else if (strstr(result_buf, "[OUTPUT_END]")) {
						hidden_index++;
						is_out = 0;
					} else if (is_hidden) {
						result_buf[strlen(result_buf) - 1] = 0;
						char *ptr1 = strchr(result_buf, ':');
						if (ptr1) {
							*ptr1 = 0;
							ptr1++;
							char *ptr2 = strchr(ptr1, ':');
							if (ptr2) {
								*ptr2 = 0;
								ptr2++;
								char *tptr = NULL;
								int x = atoi(result_buf);
								int y = atoi(ptr1);
								double vl = strtod(ptr2, &tptr);
								bayrepo_fill_outm(blob, x, y, vl);
							}
						}
					}
				}
				}
			}
			counter++;
		}
		return blob;

	}
	return NULL;
}
