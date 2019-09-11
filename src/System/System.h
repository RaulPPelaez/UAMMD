/* Raul P. Pelaez 2017. UAMMD System
   
   Contains basic information and utilities about the system UAMMD is running on.

   Provides a logger and a CPU rng

   Also provides a way to recover argc and argv from anywhere.
 */
#ifndef SYSTEM_H
#define SYSTEM_H

#include"global/defines.h"
#include"utils/utils.h"
#include"utils/cxx_utils.h"
#include"utils/debugTools.cuh"
#include"utils/parseArguments.h"
#include<cstring>
#include<ostream>
#include<iostream>
#include<cstdarg>
#include"misc/allocator.h"
#include<memory>
#include<ctime>
#include<chrono>

namespace uammd{

  using std::shared_ptr;
  using std::string;


  struct access{
    enum location{cpu, gpu, managed, nodevice};
    enum mode{read, write, readwrite, nomode};
  };
  
  struct SystemParameters{

    int device = -1;
    int cuda_arch = 0;
    int minimumCudaArch = 200;
    bool managedMemoryAvailable = false;
  };

  #ifdef MAXLOGLEVEL
  constexpr int maxLogLevel = MAXLOGLEVEL;
  #else
  constexpr int maxLogLevel = 6;
  #endif
  
  class System{
  public:
    
    using device_temporary_memory_resource = uammd::device_pool_memory_resource;
    template<class T>
    using allocator = uammd::polymorphic_allocator<T, device_temporary_memory_resource>;

  private:    
    Xorshift128plus m_rng;
    SystemParameters sysPar;
    Timer tim;
    void printWellcome();
    void printFarewell();

    int m_argc = 0;
    char ** m_argv = nullptr;
    std::shared_ptr<device_temporary_memory_resource> m_memory_resource;
    
  public:    
    System():System(0, nullptr){}
    System(int argc, char *argv[]): m_argc(argc), m_argv(argv){
      tim.tic();
      this->printWellcome();
      CudaCheckError();
      CudaSafeCall(cudaDeviceSynchronize());
      auto seed = 0xf31337Bada55D00dULL^time(NULL);
      m_rng.setSeed(seed);
      
      int dev = -1;
      size_t cuda_printf_limit;

      CudaSafeCall(cudaDeviceGetLimit(&cuda_printf_limit,cudaLimitPrintfFifoSize));
      //If the device is set from cli
      if(input_parse::parseArgument(argc, argv, "--increase_print_limit", &cuda_printf_limit)){
	log<WARNING>("[System] Setting CUDA printf buffer size to %s",
		     printUtils::prettySize(cuda_printf_limit).c_str());
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, cuda_printf_limit);
      }
      if(input_parse::parseArgument(argc, argv, "--device", &dev)){
	cudaSetDevice(dev);
      }
      else{//Otherwise set the one CUDA Driver auto sets
	CudaSafeCall(cudaFree(0));
	CudaSafeCall(cudaGetDevice(&dev));
	CudaSafeCall(cudaFree(0));
	CudaSafeCall(cudaSetDevice(dev));
      }
      CudaSafeCall(cudaFree(0));
      cudaDeviceProp deviceProp;
      CudaSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
      
      log<System::MESSAGE>("[System] Using device: %s with id: %d", deviceProp.name, dev);
      
      sysPar.device = dev;
      sysPar.cuda_arch = 100*deviceProp.major + 10*deviceProp.minor;
      if(sysPar.cuda_arch >= 600){
	sysPar.managedMemoryAvailable = true;
	log<System::DEBUG>("[System] Managed memory enabled");
      }

      log<System::MESSAGE>("[System] Compute capability of the device: %d.%d", deviceProp.major, deviceProp.minor);

      std::string line;
      fori(0,29) line += "━ ";
      log<System::MESSAGE>("%s", line.c_str());

      if(sysPar.cuda_arch < sysPar.minimumCudaArch)
	log<System::CRITICAL>("[System] Unsupported Configuration, the GPU must have at least compute capability %d (%d.%d found)", sysPar.minimumCudaArch, deviceProp.major, deviceProp.minor);
      m_memory_resource = std::make_shared<device_temporary_memory_resource>();
      CudaCheckError();

    }
    
    
    void finish(){
      CudaCheckError();
      cudaDeviceSynchronize();
      this->printFarewell();

    }
    enum level{CRITICAL=0, ERROR, WARNING, MESSAGE, STDERR, STDOUT,
	       DEBUG, DEBUG1, DEBUG2, DEBUG3, DEBUG4, DEBUG5, DEBUG6, DEBUG7};
    template<int level>
    static inline void log(char const *fmt, ...) {
      if(level<=maxLogLevel){
	if(level==CRITICAL) {
	  va_list args;
	  va_start(args, fmt);
	  fprintf(stderr, "\e[101m[CRITICAL] ");
	  vfprintf(stderr, fmt, args);
	  fprintf(stderr, "\e[0m\n");
	  exit(1);
	}
	if(level==ERROR){
	  va_list args;
	  va_start(args, fmt);
	  fprintf(stderr, "\e[91m[ ERROR ] \e[0m");
	  vfprintf(stderr, fmt, args);
	  fprintf(stderr, "\n");
	}
	if(level==WARNING){
	  va_list args;
	  va_start(args, fmt);
	  fprintf(stderr, "\e[93m[WARNING] \e[0m");
	  vfprintf(stderr, fmt, args);
	  fprintf(stderr, "\n");
	}
	if(level==MESSAGE){
	  va_list args;
	  va_start(args, fmt);
	  fprintf(stderr, "\e[92m[MESSAGE] \e[0m");
	  vfprintf(stderr, fmt, args);
	  fprintf(stderr, "\n");    
	}    
	if(level==STDERR){
	  va_list args;
	  va_start(args, fmt);
	  vfprintf(stderr, fmt, args);
	  fprintf(stderr, "\n");
	}

	if(level==STDOUT){
	  va_list args;
	  va_start(args, fmt);
	  vfprintf(stdout, fmt, args);
	  fprintf(stdout, "\n");
	}
	if(level==DEBUG){
	  va_list args;
	  va_start(args, fmt);
	  fprintf(stderr, "\e[96m[ DEBUG ] \e[0m");
	  vfprintf(stderr, fmt, args);
	  fprintf(stderr, "\n");     
	}
	if(level>=DEBUG1 && level<=DEBUG7){
	  va_list args;
	  va_start(args, fmt);
	  fprintf(stderr, "\e[96m[ DEBUG ] \e[0m");
	  vfprintf(stderr, fmt, args);
	  fprintf(stderr, "\n");     
	}
      }
    }  

    Xorshift128plus& rng(){ return m_rng;}

    int getargc(){ return this->m_argc;}
    const char ** getargv(){return (const char **)this->m_argv;}

    const SystemParameters getSystemParameters(){
      return sysPar;
    }

    template<class T>
    allocator<T> getTemporaryDeviceAllocator(){
      return allocator<T>(m_memory_resource.get());
    }
    
  };




  void System::printWellcome(){
    std::string separator;
    fori(0,60) separator += "━";
    separator += "┓";
    log<System::MESSAGE>("%s", separator.c_str());
    
    string line1 = "\033[94m╻\033[0m \033[94m╻┏━┓┏┳┓┏┳┓╺┳┓\033[0m";
    string line2 = "\033[94m┃\033[0m \033[94m┃┣━┫┃┃\033[34m┃┃┃┃\033[0m \033[34m┃┃\033[0m";
    string line3 = "\033[34m┗━┛╹\033[0m \033[34m╹╹\033[0m \033[34m╹╹\033[0m \033[34m╹╺┻┛\033[0m";
    log<System::MESSAGE>("%s",line1.c_str());
    log<System::MESSAGE>("%s Version: %s", line2.c_str(), UAMMD_VERSION);
    log<System::MESSAGE>("%s",line3.c_str());
    log<System::MESSAGE>("Compiled at: %s %s", __DATE__, __TIME__);
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log<System::MESSAGE>("Computation started at %s", std::ctime(&time));


  }


  void System::printFarewell(){
    cudaDeviceSynchronize();
    std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log<System::MESSAGE>("Computation finished at %s",std::ctime(&time));
    log<System::MESSAGE>("Time elapsed since creation: %fs", tim.toc());
    std::string line;
    fori(0,60) line +="━";
    line +="┛";

    log<System::MESSAGE>("%s", line.c_str());
  }
  
  
}
#endif
