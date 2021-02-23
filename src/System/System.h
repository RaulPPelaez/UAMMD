// Raul P. Pelaez 2019. UAMMD System
#ifndef UAMMD_SYSTEM_H
#define UAMMD_SYSTEM_H

#include"global/defines.h"
#include"utils/utils.h"
#include"utils/exception.h"
#include"utils/cxx_utils.h"
#include"utils/debugTools.h"
#include"utils/parseArguments.h"
#include"System/Log.h"
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

  class insuficient_compute_capability_exception: public uammd::exception{
    const char* what() const noexcept {
      return "Insuficient compute capability";
    }
  };

  class System{
  public:
#ifndef UAMMD_DEBUG
    using resource = uammd::managed_memory_resource;
#else
    using resource = uammd::device_memory_resource;
#endif
    using device_temporary_memory_resource = uammd::pool_memory_resource_adaptor<resource>;
    template<class T>
    using allocator_thrust = uammd::polymorphic_allocator<T, device_temporary_memory_resource,
							  thrust::cuda::pointer<T>>;
    template<class T>
    using allocator = uammd::polymorphic_allocator<T , device_temporary_memory_resource>;

    enum LogLevel{CRITICAL=0, ERROR, EXCEPTION, WARNING, MESSAGE, STDERR, STDOUT,
		  DEBUG, DEBUG1, DEBUG2, DEBUG3, DEBUG4, DEBUG5, DEBUG6, DEBUG7};

  private:
    struct CommandLineOptions{
      int device = -1;
      size_t cuda_printf_limit = 0;
    };

    Xorshift128plus m_rng;
    SystemParameters sysPar;
    Timer tim;

    void printWelcome();
    void printFarewell();

    int m_argc = 0;
    char ** m_argv = nullptr;

    CommandLineOptions processInputArguments(){
      log<DEBUG1>("[System] Reading command line arguments");
      try{
	return tryToProcessInputArguments();
      }
      catch(...){
	std::throw_with_nested(std::runtime_error("processInputArguments failed"));
      }
    }

    CommandLineOptions tryToProcessInputArguments(){
      CommandLineOptions op;
      CommandLineArgumentParser cmd(m_argc, m_argv);
      std::string flag;
      flag = "--device";
      if(cmd.isFlagPresent(flag))
	op.device = cmd.getFlagArgument<int>(flag);
      flag = "--increase_print_limit";
      if(cmd.isFlagPresent(flag))
	op.cuda_printf_limit = cmd.getFlagArgument<size_t>(flag);
      return op;
    }

    void initializeCUDA(){
      try{
	if(sysPar.device>=0){
	  CudaSafeCall(cudaSetDevice(sysPar.device));
	}
	CudaSafeCall(cudaFree(0));
	CudaSafeCall(cudaDeviceSynchronize());
	CudaCheckError();
      }
      catch(...){
	log<System::ERROR>("[System] Exception raised at CUDA initialization");
	throw;
      }
      log<System::MESSAGE>("[System] CUDA initialized");
    }

    void storeComputeCapability(){
      cudaDeviceProp deviceProp;
      CudaSafeCall(cudaGetDeviceProperties(&deviceProp, sysPar.device));
      log<System::MESSAGE>("[System] Using device: %s with id: %d",
			   deviceProp.name, sysPar.device);
      sysPar.cuda_arch = 100*deviceProp.major + 10*deviceProp.minor;
      if(sysPar.cuda_arch >= 600){
	sysPar.managedMemoryAvailable = true;
	log<System::DEBUG>("[System] Managed memory enabled");
      }
      log<System::MESSAGE>("[System] Compute capability of the device: %d.%d",
			   deviceProp.major, deviceProp.minor);
      if(sysPar.cuda_arch < sysPar.minimumCudaArch){
	log<System::ERROR>("[System] Unsupported Configuration, the GPU must have at least compute capability %d (%d.%d found)",
			   sysPar.minimumCudaArch, deviceProp.major, deviceProp.minor);
	throw insuficient_compute_capability_exception();
      }


    }

  public:
    System():System(0, nullptr){}
    System(int argc, char *argv[]): m_argc(argc), m_argv(argv){
      tim.tic();
      this->printWelcome();
      auto seed = 0xf31337Bada55D00dULL^time(NULL);
      m_rng.setSeed(seed);
      auto options = processInputArguments();
      sysPar.device = -1;
      if(options.device>=0){
	sysPar.device = options.device;
      }
      this->initializeCUDA();
      if(options.device<0){
	CudaSafeCall(cudaGetDevice(&(sysPar.device)));
      }
      if(options.cuda_printf_limit > 0){
	CudaSafeCall(cudaDeviceSetLimit(cudaLimitPrintfFifoSize,
					options.cuda_printf_limit));
      }
      this->storeComputeCapability();
      std::string line;
      fori(0,29) line += "━ ";
      log<System::MESSAGE>("%s", line.c_str());
      CudaCheckError();
    }

    void finish(){
      log<DEBUG2>("[System] finish");
      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
      detail::get_default_resource<device_temporary_memory_resource>()->free_all();
      this->printFarewell();
    }

    template<int level, class ...T>
    static inline void log(char const *fmt, T... args){
      Logging::log<level>(fmt, args...);
      if(level == CRITICAL){
	throw std::runtime_error("System encountered an unrecoverable error");
      }
    }
    template<int level>
    static inline void log(const std::string &msg){
      log<level>("%s", msg.c_str());
    }

    Xorshift128plus& rng(){ return m_rng;}

    int getargc() const{ return this->m_argc;}
    const char ** getargv() const{return (const char **)this->m_argv;}

    const SystemParameters getSystemParameters() const{
      return sysPar;
    }

    template<class T = char>
    static allocator<T> getTemporaryDeviceAllocator(){
      return allocator<T>();
    }

  };

  void System::printWelcome(){
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
#ifdef DOUBLE_PRECISION
    log<System::MESSAGE>("Compiled in double precision mode");
#else
    log<System::MESSAGE>("Compiled in single precision mode");
#endif
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
