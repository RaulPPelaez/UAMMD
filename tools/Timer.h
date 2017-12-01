#ifndef TIMER_H
#define TIMER_H
//#include<chrono>
#include<sys/time.h>



class Timer{
public:
  Timer(){}
  void tic();
  float toc();
private:
  //std::chrono::time_point<std::chrono::system_clock> t;
  struct timeval start, end;
};

#endif
