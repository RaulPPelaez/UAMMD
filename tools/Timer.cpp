#include "Timer.h"

void Timer::tic(){
  //this->t = std::chrono::system_clock::now();
  gettimeofday(&start, 0);
}

float Timer::toc(){
  //  auto t2 = std::chrono::system_clock::now();
  gettimeofday(&end, 0);
  //return std::chrono::duration<float>(t2 - t).count();
  return ((end.tv_sec  - start.tv_sec) * 1000000u + 
	  end.tv_usec - start.tv_usec) / 1.e6;
}

