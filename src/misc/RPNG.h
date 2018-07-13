#ifndef RPNG_H
#define RPNG_H
#include<png.h>
#include<stdlib.h>
#include<string.h>


bool savePNG(const char* fileName,
	     unsigned char* px, int wx, int wy,
	     int ctype = PNG_COLOR_TYPE_RGB_ALPHA);

#endif
