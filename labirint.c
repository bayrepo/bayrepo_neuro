/*
 * labirint.c
 *
 *  Created on: Nov 1, 2019
 *      Author: alexey
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "neuro_web_client.h"

#define NOT_ENOUGH_OF_MEMORY(x) if (!x) { \
		printf("Not enough memory\n"); \
		exit(255); \
	}

#define HERO 1
#define FINISH 2
#define BLOCK 3
#define WASHERE 4

void draw_screen(int size, char *arena) {
	int x, y;
	for (y = 0; y < (size + 2); y++) {
		for (x = 0; x < (size + 2); x++) {
			if (x == 0 || x == (size + 1)) {
				mvprintw(y, x, "#");
			} else if (y == 0 || y == (size + 1)) {
				mvprintw(y, x, "#");
			} else if (arena[(y - 1) * size + x - 1] == HERO) {
				mvprintw(y, x, "&");
			} else if (arena[(y - 1) * size + x - 1] == FINISH) {
				mvprintw(y, x, "7");
			} else {
				mvprintw(y, x, " ");
			}
		}
	}
}

double is_new_pos_blocked(int x, int y, int size, char *arena) {
	if (x < 0 || y < 0)
		return 1.0;
	if (x >= size || y >= size)
		return 1.0;
	if (arena[x * size + y] == BLOCK)
		return 1.0;
	return 0.0;
}

double was_here(int x, int y, int size, char *arena) {
	if (x < 0 || y < 0)
		return 1.0;
	if (x >= size || y >= size)
		return 1.0;
	if (arena[x * size + y] == WASHERE)
		return 1.0;
	return 0.0;
}

void smallest_ways(int x_cur, int y_cur, int finish_x, int finish_y,
		double *left, double *right, double *up, double *down) {
	int direct[4] = { 0, 0, 0, 0 };

	*left = 0.0;
	*right = 0.0;
	*up = 0.0;
	*down = 0.0;

	int dxl = x_cur - 1 - finish_x;
	int dxr = x_cur + 1 - finish_x;
	int dyu = y_cur - 1 - finish_y;
	int dyd = y_cur + 1 - finish_y;

	int dx_o = x_cur - finish_x;
	int dy_o = y_cur - finish_y;

	direct[0] = dxl * dxl + dy_o * dy_o; //left
	direct[1] = dxr * dxr + dy_o * dy_o; //right
	direct[2] = dx_o * dx_o + dyu * dyu; //up
	direct[4] = dx_o * dx_o + dyd * dyd; //down;

	int min = direct[0];
	int index = 0;
	for (index = 1; index < 4; index++) {
		if (direct[index] <= min)
			min = direct[index];
	}
	for (index = 0; index < 4; index++) {
		if (direct[index] == min) {
			switch (index) {
			case 0:
				*left = 1.0;
				break;
			case 1:
				*right = 1.0;
				break;
			case 2:
				*up = 1.0;
				break;
			case 3:
				*down = 1.0;
				break;
			default:
				break;
			}
		}
	}
}

#define NET 3

int direction_result(double *data, int size) {
	int index = 0;
	double max = data[0];
	for (index = 1; index < size; index++) {
		if (data[index] > max)
			max = data[index];
	}

	for (index = 0; index < size; index++) {
		if (data[index] == max)
			return index;
	}
	return 0;
}

int main(int argc, char* argv[]) {
	double inputs[12];
	double outputs[4];
	char curl_error[1024];

	srand(time(NULL));
	if (argc <= 1) {
		printf("enter size of labirint\n");
		exit(255);
	}
	int size = atoi(argv[1]);
	if (size < 10) {
		printf("enter size more than 9\n");
		exit(255);
	}
	char *arena = calloc(1, size * size);
	NOT_ENOUGH_OF_MEMORY(arena);
	int startX = rand() % size;
	int startY = rand() % size;

	int endX = rand() % size;
	int endY = rand() % size;

	while (startX == endX && startY == endY) {
		endX = rand() % size;
		endY = rand() % size;
	}

	arena[startY * size + startX] = HERO;
	arena[endY * size + endX] = FINISH;

	int newX, newY;

	web_client_init("http://127.0.0.1:10111");

	initscr();
	curs_set(0);
	noecho();

	while (1) {
		draw_screen(size, arena);

		newX = startX;
		newY = startY;

		inputs[0] = is_new_pos_blocked(startX + 1, startY, size, arena); //right
		inputs[1] = is_new_pos_blocked(startX - 1, startY, size, arena); //left
		inputs[2] = is_new_pos_blocked(startX, startY - 1, size, arena); //up
		inputs[3] = is_new_pos_blocked(startX, startY + 1, size, arena); //down

		inputs[4] = was_here(startX + 1, startY, size, arena); //right
		inputs[5] = was_here(startX - 1, startY, size, arena); //left
		inputs[6] = was_here(startX, startY - 1, size, arena); //up
		inputs[7] = was_here(startX, startY + 1, size, arena); //down

		smallest_ways(startX, startY, endX, endY, &inputs[9], &inputs[8],
				&inputs[10], &inputs[11]);

		web_send_inputs_to_net(NET, (double *) &inputs, 12, (double *) &outputs,
				4, curl_error);
		int direct = direction_result((double *) outputs, 4);
		switch (direct) {
		case 0:
			newX--;
			break;
		case 1:
			newX++;
			break;
		case 2:
			newY--;
			break;
		case 3:
			newY++;
			break;
		default:
			break;
		}
		mvprintw(newY + 1, newX + 1, "@");

		mvprintw(size + 2, 0, "Correct?(y/a/w/d/s/g/q):");
		refresh();
		int ch = getch();
		if (ch == 'q')
			break;
		if (ch == 'y') {
			web_send_train_to_net(NET, (double *) &inputs, 12,
					(double *) &outputs, 4, curl_error);
			arena[startY * size + startX] = WASHERE;
			arena[newY * size + newX] = HERO;
			startX = newX;
			startY = newY;
		}
		if (ch == 'g') {
			arena[startY * size + startX] = WASHERE;
			arena[newY * size + newX] = HERO;
			startX = newX;
			startY = newY;
		}
		if (ch == 'a') {
			outputs[0] = 0.0;
			outputs[1] = 1.0;
			outputs[2] = 0.0;
			outputs[3] = 0.0;
			newX = startX - 1;
			newY = startY;
			web_send_train_to_net(NET, (double *) &inputs, 12,
					(double *) &outputs, 4, curl_error);
			arena[startY * size + startX] = WASHERE;
			arena[newY * size + newX] = HERO;
			startX = newX;
			startY = newY;
		}
		if (ch == 'w') {
			outputs[0] = 0.0;
			outputs[1] = 0.0;
			outputs[2] = 1.0;
			outputs[3] = 0.0;
			newX = startX;
			newY = startY - 1;
			web_send_train_to_net(NET, (double *) &inputs, 12,
					(double *) &outputs, 4, curl_error);
			arena[startY * size + startX] = WASHERE;
			arena[newY * size + newX] = HERO;
			startX = newX;
			startY = newY;
		}
		if (ch == 'd') {
			outputs[0] = 1.0;
			outputs[1] = 0.0;
			outputs[2] = 0.0;
			outputs[3] = 0.0;
			newX = startX + 1;
			newY = startY;
			web_send_train_to_net(NET, (double *) &inputs, 12,
					(double *) &outputs, 4, curl_error);
			arena[startY * size + startX] = WASHERE;
			arena[newY * size + newX] = HERO;
			startX = newX;
			startY = newY;
		}
		if (ch == 's') {
			outputs[0] = 0.0;
			outputs[1] = 0.0;
			outputs[2] = 0.0;
			outputs[3] = 1.0;
			newX = startX;
			newY = startY + 1;
			web_send_train_to_net(NET, (double *) &inputs, 12,
					(double *) &outputs, 4, curl_error);
			arena[startY * size + startX] = WASHERE;
			arena[newY * size + newX] = HERO;
			startX = newX;
			startY = newY;
		}
	}

	endwin();
	web_client_clean();
	free(arena);
	return 0;

	return 0;
}
