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

#include "arena_cycle.h"

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
			} else if (arena[(y - 1) * size + x - 1] == WASHERE) {
				mvprintw(y, x, "+");
			} else if (arena[(y - 1) * size + x - 1] == BLOCK) {
				mvprintw(y, x, "*");
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
	if (arena[y * size + x] == BLOCK)
		return 1.0;
	return 0.0;
}

double was_here(int x, int y, int size, char *arena) {
	if (x < 0 || y < 0)
		return 1.0;
	if (x >= size || y >= size)
		return 1.0;
	if (arena[y * size + x] == WASHERE || arena[y * size + x] == HERO)
		return 1.0;
	return 0.0;
}

void smallest_ways(int x_cur, int y_cur, int finish_x, int finish_y,
		double *left, double *right, double *up, double *down, int size) {
	int direct[4] = { 0, 0, 0, 0 };

	*left = 0.0;
	*right = 0.0;
	*up = 0.0;
	*down = 0.0;

	int dxl = x_cur - 1 - finish_x;
	int dxr = x_cur + 1 - finish_x;
	int dyu = y_cur - 1 - finish_y;
	int dyd = y_cur + 1 - finish_y;
	mvprintw(size + 6, 0, "DXL%d DXR%d DYU%d DYD%d", dxl, dxr, dyu, dyd);

	int dx_o = x_cur - finish_x;
	int dy_o = y_cur - finish_y;
	mvprintw(size + 7, 0, "DXO%d DYO%d", dx_o, dy_o);

	direct[0] = dxl * dxl + dy_o * dy_o; //left
	direct[1] = dxr * dxr + dy_o * dy_o; //right
	direct[2] = dx_o * dx_o + dyu * dyu; //up
	direct[3] = dx_o * dx_o + dyd * dyd; //down;

	mvprintw(size + 8, 0, "SL%d SR%d SU%d SD%d", direct[0], direct[1],
			direct[2], direct[3]);

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

#define NET 5

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

void makeWalls(int size, char *arena) {
	int numberOfWalls = rand() % 20 + 5;
	int index;
	while (numberOfWalls) {
		int wSize = rand() % 5 + 2;
		int x = rand() % size;
		int y = rand() % size;
		int direct = rand() % 2;
		if (direct) {
			for (index = 0; index < wSize; index++) {
				if (arena[y * size + x] == 0)
					arena[y * size + x] = BLOCK;
				x++;
				if (x >= size)
					break;
			}
		} else {
			for (index = 0; index < wSize; index++) {
				if (arena[y * size + x] == 0)
					arena[y * size + x] = BLOCK;
				y++;
				if (y >= size)
					break;
			}
		}
		numberOfWalls--;
	}
}

int foundWay(int size, char *arena, int x_cur, int y_cur, int finish_x,
		int finish_y, char *way, int counter, int old_direct, char **copy_arena) {
	if (!*copy_arena) {
		*copy_arena = calloc(1, size * size);
		NOT_ENOUGH_OF_MEMORY(*copy_arena);
		memcpy(*copy_arena, arena, size * size);
	}
	int direct[4] = { 0, 0, 0, 0 };
	int is_checked[4] = { 0, 0, 0, 0 };
	int result = 0;

	char *ar = *copy_arena;

	if (x_cur
			< 0|| y_cur < 0 || x_cur>=size || y_cur >= size || arena[y_cur * size + x_cur] == BLOCK
			|| ar[y_cur * size + x_cur] == WASHERE) {
		return 0;
	}

	if (ar[y_cur * size + x_cur] != HERO) {
		ar[y_cur * size + x_cur] = WASHERE;
		//if (arena[y_cur * size + x_cur] == 0){
		//	arena[y_cur * size + x_cur]=WASHERE;
		//}
	}

	if (old_direct >= 0) {
		is_checked[old_direct] = 1;
	}

	int dxl = x_cur - 1 - finish_x;
	int dxr = x_cur + 1 - finish_x;
	int dyu = y_cur - 1 - finish_y;
	int dyd = y_cur + 1 - finish_y;

	int dx_o = x_cur - finish_x;
	int dy_o = y_cur - finish_y;

	direct[0] = dxl * dxl + dy_o * dy_o; //left
	direct[1] = dxr * dxr + dy_o * dy_o; //right
	direct[2] = dx_o * dx_o + dyu * dyu; //up
	direct[3] = dx_o * dx_o + dyd * dyd; //down;

	int index = 0;
	for (index = 0; index < 4; index++) {
		int newX = x_cur;
		int newY = y_cur;
		int min = size * size + 1;
		int jndex = 0;
		int pos = 0;
		for (jndex = 0; jndex < 4; jndex++) {
			if (!is_checked[jndex] && direct[jndex] <= min) {
				min = direct[jndex];
				pos = jndex;
			}
		}
		is_checked[pos] = 1;
		char ch = 'a';
		int old = -1;

		switch (pos) {
		case 0:
			newX--;
			ch = 'a';
			old = 1;
			break;
		case 1:
			newX++;
			ch = 'd';
			old = 0;
			break;
		case 2:
			newY--;
			ch = 'w';
			old = 3;
			break;
		case 3:
			newY++;
			ch = 's';
			old = 2;
			break;
		default:
			break;
		}
		//draw_screen(size, arena);
		//mvprintw(newY + 1, newX + 1, "@");
		//refresh();
		//sleep(1);

		if ((newX == finish_x) && (newY == finish_y)) {
			way[counter] = ch;
			return 1;
		}

		if (ar[newY * size + newX] == HERO) {
			return 0;
		}

		result = foundWay(size, arena, newX, newY, finish_x, finish_y, way,
				counter + 1, old, copy_arena);
		if (result) {
			way[counter] = ch;
			break;
		}
	}

	return result;
}

void printArena(int size, char *arena) {
	FILE *fp = fopen("arena.h", "w");
	if (fp) {
		fprintf(fp, "char arena_dat[%d][%d]={\n", size, size);
		int index = 0;
		int jndex = 0;
		for (index = 0; index < size; index++) {
			fprintf(fp, "{ ");
			for (jndex = 0; jndex < size; jndex++) {
				fprintf(fp, "%d%s ", arena[index * size + jndex],
						(jndex == (size - 1) ? "" : ","));
			}
			fprintf(fp, "}%s\n", (index == (size - 1) ? "" : ","));
		}
		fprintf(fp, "};\n");
		fclose(fp);
	}
}

void restoreArena(char *arena) {
	int size = 20;
	char *ar = (char *) &arena_dat;
	int index = 0;
	int jndex = 0;
	for (index = 0; index < size; index++) {
		for (jndex = 0; jndex < size; jndex++) {
			arena[index * size + jndex] = ar[index * size + jndex];
		}
	}
}

void findSymbol(char *arena, char symb, int *x, int *y) {
	int size = 20;
	int index = 0;
	int jndex = 0;
	for (index = 0; index < size; index++) {
		for (jndex = 0; jndex < size; jndex++) {
			if (arena[index * size + jndex] == symb) {
				*x = jndex;
				*y = index;
				return;
			}
		}
	}
	return;
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
	char *way = calloc(1, size * size);
	NOT_ENOUGH_OF_MEMORY(way);
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

	makeWalls(size, arena);

	//printArena(size, arena);
	//restoreArena(arena);
	//findSymbol(arena, HERO, &startX, &startY);
	//findSymbol(arena, FINISH, &endX, &endY);

	int newX, newY;

	web_client_init("http://127.0.0.1:10111");

	initscr();
	curs_set(0);
	noecho();

	int step = 0;
	draw_screen(size, arena);
	refresh();
	char *tmp = NULL;
	if (!foundWay(size, arena, startX, startY, endX, endY, way, 0, -1, &tmp)) {
		mvprintw(size + 2, 0, "No way");
		getch();
		exit(255);
	}
	free(tmp);

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
				&inputs[10], &inputs[11], size);

		//web_send_inputs_to_net(NET, (double *) &inputs, 12, (double *) &outputs,
		//		4, curl_error);
		int direct = direction_result((double *) outputs, 4);
		switch (direct) {
		case 0:
			newX++;
			break;
		case 1:
			newX--;
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
		mvprintw(size + 3, 0, "BR%fBL%fBU%fBD%f", inputs[0], inputs[1],
				inputs[2], inputs[3]);
		mvprintw(size + 4, 0, "WR%fWL%fWU%fWD%f", inputs[4], inputs[5],
				inputs[6], inputs[7]);
		mvprintw(size + 5, 0, "DR%fDL%fDU%fDD%f", inputs[8], inputs[9],
				inputs[10], inputs[11]);
		refresh();
		//int ch = getch();
		int ch = 'g';
		if (!way[step])
			ch = 'q';
		else {
			ch = way[step++];
		}
		sleep(1);

		if ((startX == endX) && (startY == endY))
			ch = 'q';
		//getch();

		if (ch == 'q')
			break;
		if (ch == 'y') {
			//web_send_train_to_net(NET, (double *) &inputs, 12,
			//		(double *) &outputs, 4, curl_error);
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
			//web_send_train_to_net(NET, (double *) &inputs, 12,
			//		(double *) &outputs, 4, curl_error);
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
			//web_send_train_to_net(NET, (double *) &inputs, 12,
			//		(double *) &outputs, 4, curl_error);
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
			//web_send_train_to_net(NET, (double *) &inputs, 12,
			//		(double *) &outputs, 4, curl_error);
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
			//web_send_train_to_net(NET, (double *) &inputs, 12,
			//		(double *) &outputs, 4, curl_error);
			arena[startY * size + startX] = WASHERE;
			arena[newY * size + newX] = HERO;
			startX = newX;
			startY = newY;
		}
	}

	endwin();
	web_client_clean();
	free(arena);
	free(way);
	return 0;

	return 0;
}
