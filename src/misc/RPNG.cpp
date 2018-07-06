#include"RPNG.h"



void png_user_warn(png_structp ctx, png_const_charp str){
  fprintf(stderr, "libpng: warning: %s\n", str);
}

void png_user_error(png_structp ctx, png_const_charp str){
  fprintf(stderr, "libpng: error: %s\n", str);
}

bool savePNG(const char* fileName,
	     unsigned char* px, int wx, int wy,
	     int ctype){

  FILE *fp;
  png_structp png_ptr;
  png_infop info_ptr;
  
  png_bytep *row_pointers;

  fp = fopen(fileName, "wb");

  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 
				    NULL, png_user_error, png_user_warn);
  info_ptr = png_create_info_struct(png_ptr);



  if (png_ptr == NULL) {
    printf("png_create_write_struct error!\n");
    return -1;
  }


  if (info_ptr == NULL) {
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    printf("png_create_info_struct error!\n");
    exit(-1);
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    exit(-1);
  }


  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr, wx, wy, 8, ctype, PNG_INTERLACE_NONE,
	       PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  png_set_packing(png_ptr);

  row_pointers = new png_bytep[wx];
  

  for(int i=0; i<wx; i++) row_pointers[i] = (png_bytep)px+(wy-i-1)*wx*4;
  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, info_ptr);

  delete row_pointers;

  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
  return true;
}
