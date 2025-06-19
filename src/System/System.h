// Raul P. Pelaez 2019-2025. UAMMD System
// clang-format off
/**
 * @file System.h
 * @brief Core system configuration and initialization module for UAMMD.
 *
 * System is UAMMD's core module responsible for interfacing with the
 * machine the software is running on. It handles things like CUDA
 * initialization, command line argument parsing, and provides a logging system.
 * System is the first module to be initialized when UAMMD starts, and the last
 * one to be finalized.
 * @note System is automatically created by @ref uammd::ParticleData "ParticleData" when not provided by the user. Most times you will not need to create a @ref uammd::System "System" object manually.
 */
// clang-format on
#ifndef UAMMD_SYSTEM_H
#define UAMMD_SYSTEM_H

#include "System/Log.h"
#include "global/defines.h"
#include "misc/allocator.h"
#include "utils/debugTools.h"
#include "utils/exception.h"
#include "utils/parseArguments.h"
#include "utils/utils.h"
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <memory>

namespace uammd {

using std::shared_ptr;
using std::string;

/**
 * @struct SystemParameters
 * @brief Holds system parameters for the UAMMD simulation.
 */
struct SystemParameters {
  int device = -1;   ///< The CUDA device ID to use for the simulation.
  int cuda_arch = 0; ///< The compute capability of the CUDA device, represented
                     ///< as an integer (e.g., 600 for 6.0).
  int minimumCudaArch = 200;
  bool managedMemoryAvailable = false;
};

class insuficient_compute_capability_exception : public uammd::exception {
  const char *what() const noexcept { return "Insuficient compute capability"; }
};

/**
 * @class System
 * @brief The central runtime manager for CUDA-based UAMMD simulations.
 * @note Most of the time you do not need to handle the creation of ::System.
 * ::ParticleData will auto initialize it for you if you do not provide one. You
 * can request ::ParticleData for a reference to ::System with
 * ::ParticleData::getSystem().
 * ::System should be created explicitly if the user wants to provide command
 line arguments.

 */
class System {
public:
#ifndef UAMMD_DEBUG
  using resource = uammd::managed_memory_resource;
#else
  using resource = uammd::device_memory_resource;
#endif
  using device_temporary_memory_resource =
      uammd::pool_memory_resource_adaptor<resource>;
  template <class T>
  using allocator_thrust =
      uammd::polymorphic_allocator<T, device_temporary_memory_resource,
                                   thrust::cuda::pointer<T>>;
  template <class T>
  using allocator =
      uammd::polymorphic_allocator<T, device_temporary_memory_resource>;

  /**
   * @brief Defines the logging levels used in UAMMD.
   * \private
   */
  enum LogLevel {
    CRITICAL = 0,
    ERROR,
    EXCEPTION,
    WARNING,
    MESSAGE,
    STDERR,
    STDOUT,
    DEBUG,
    DEBUG1,
    DEBUG2,
    DEBUG3,
    DEBUG4,
    DEBUG5,
    DEBUG6,
    DEBUG7
  };

private:
  struct CommandLineOptions {
    int device = -1;
    size_t cuda_printf_limit = 0;
  };

  Xorshift128plus m_rng;
  SystemParameters sysPar;
  Timer tim;

  void printWelcome();
  void printFarewell();

  int m_argc = 0;
  char **m_argv = nullptr;

  CommandLineOptions processInputArguments() {
    log<DEBUG1>("[System] Reading command line arguments");
    try {
      return tryToProcessInputArguments();
    } catch (...) {
      std::throw_with_nested(
          std::runtime_error("processInputArguments failed"));
    }
  }

  CommandLineOptions tryToProcessInputArguments() {
    CommandLineOptions op;
    CommandLineArgumentParser cmd(m_argc, m_argv);
    std::string flag;
    flag = "--device";
    if (cmd.isFlagPresent(flag))
      op.device = cmd.getFlagArgument<int>(flag);
    flag = "--increase_print_limit";
    if (cmd.isFlagPresent(flag))
      op.cuda_printf_limit = cmd.getFlagArgument<size_t>(flag);
    return op;
  }

  void initializeCUDA() {
    try {
      if (sysPar.device >= 0) {
        CudaSafeCall(cudaSetDevice(sysPar.device));
      }
      CudaSafeCall(cudaFree(0));
      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
    } catch (...) {
      log<System::ERROR>("[System] Exception raised at CUDA initialization");
      throw;
    }
    log<System::MESSAGE>("[System] CUDA initialized");
  }

  void storeComputeCapability() {
    cudaDeviceProp deviceProp;
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, sysPar.device));
    log<System::MESSAGE>("[System] Using device: %s with id: %d",
                         deviceProp.name, sysPar.device);
    sysPar.cuda_arch = 100 * deviceProp.major + 10 * deviceProp.minor;
    if (sysPar.cuda_arch >= 600) {
      sysPar.managedMemoryAvailable = true;
      log<System::DEBUG>("[System] Managed memory enabled");
    }
    log<System::MESSAGE>("[System] Compute capability of the device: %d.%d",
                         deviceProp.major, deviceProp.minor);
    if (sysPar.cuda_arch < sysPar.minimumCudaArch) {
      log<System::ERROR>("[System] Unsupported Configuration, the GPU must "
                         "have at least compute capability %d (%d.%d found)",
                         sysPar.minimumCudaArch, deviceProp.major,
                         deviceProp.minor);
      throw insuficient_compute_capability_exception();
    }
  }

public:
  /**
   * @brief Construct a System object with no command-line arguments.
   *
   * This initializes the system with default values, prepares the CUDA backend,
   * sets up logging, seeds the random number generator, and performs
   * device capability checks.
   */
  System() : System(0, nullptr) {}
  /**
   * @brief Construct a System object using command-line arguments.
   *
   * Parses `argv` to extract optional parameters like `--device`.  Initializes
   * the CUDA environment accordingly, performs logging setup, seeds the RNG,
   * and checks device compatibility.
   *
   * @param argc Argument count.
   * @param argv Argument vector.
   */
  System(int argc, char *argv[]) : m_argc(argc), m_argv(argv) {
    tim.tic();
    this->printWelcome();
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    auto seed = 0xf31337Bada55D00dULL ^ nanoseconds;
    m_rng.setSeed(seed);
    auto options = processInputArguments();
    sysPar.device = -1;
    if (options.device >= 0) {
      sysPar.device = options.device;
    }
    this->initializeCUDA();
    if (options.device < 0) {
      CudaSafeCall(cudaGetDevice(&(sysPar.device)));
    }
    if (options.cuda_printf_limit > 0) {
      CudaSafeCall(cudaDeviceSetLimit(cudaLimitPrintfFifoSize,
                                      options.cuda_printf_limit));
    }
    this->storeComputeCapability();
    std::string line;
    fori(0, 29) line += "━ ";
    log<System::MESSAGE>("%s", line.c_str());
    CudaCheckError();
  }

  /**
   * @brief Destructor for System.
   *
   * Cleans up resources, synchronizes the device, and prints a farewell
   * message.
   */
  void finish() {
    log<DEBUG2>("[System] finish");
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
    if (get_default_resource<device_temporary_memory_resource>()
            ->has_allocated_blocks()) {
      log<System::WARNING>(
          "[System] System::finish was called but UAMMD-controlled resources "
          "were found. This will probably result in a crash.");
    }
    get_default_resource<device_temporary_memory_resource>()->free_all();
    this->printFarewell();
  }

  /**
   * @brief Log a message at a specific log level.
   *
   * @verbatim embed:rst:leading-asterisk
   * See the :ref:`logging-doc` section for more details on logging levels.
   * @endverbatim
   */
  template <int level, class... T>
  static inline void log(char const *fmt, T... args) {
    Logging::log<level>(fmt, args...);
    if (level == CRITICAL) {
      throw std::runtime_error("System encountered an unrecoverable error");
    }
  }
  /**
   * @brief Log a message at a specific log level using a string.
   *
   * This is a convenience function that allows logging messages using a
   * std::string instead of a format string.
   */
  template <int level> static inline void log(const std::string &msg) {
    log<level>("%s", msg.c_str());
  }

  /**
   * @brief Get the random number generator used by the system.
   *
   * This returns a reference to the Xorshift128plus random number generator
   * used throughout UAMMD for generating random numbers (or seeding other
   * generators).
   */
  Xorshift128plus &rng() { return m_rng; }

  /**
   * @brief Get the number of command-line arguments System was initialized
   * with.
   *
   */
  int getargc() const { return this->m_argc; }
  /**
   * @brief Get the command-line arguments System was initialized with.
   *
   * This returns a pointer to the array of command-line arguments. The
   * arguments are stored as `char*` pointers, so you can access them as C-style
   * strings.
   */
  const char **getargv() const { return (const char **)this->m_argv; }

  /**
   * @brief Get the system parameters.
   *
   * This returns a reference to the SystemParameters struct, which contains
   * information about the CUDA device, compute capability, and managed memory
   * availability.
   */
  const SystemParameters getSystemParameters() const { return sysPar; }

  /**
   * @brief Get the default device memory resource.
   *
   * This returns an cached CUDA memory allocator that can be used to quickly
   * allocate temporary memory.
   * @verbatim embed:rst:leading-asterisk
   * See the :ref:`memory-management-doc` section for more details on memory
   * management in UAMMD.
   * @endverbatim
   * @tparam T The type of the allocator, defaults to char.
   * @return An allocator that can be used to allocate memory on the device.
   */
  template <class T = char> static allocator<T> getTemporaryDeviceAllocator() {
    return allocator<T>();
  }
};

void System::printWelcome() {
  std::string separator;
  fori(0, 60) separator += "━";
  separator += "┓";
  log<System::MESSAGE>("%s", separator.c_str());
  string line1 = "\033[94m╻\033[0m \033[94m╻┏━┓┏┳┓┏┳┓╺┳┓\033[0m";
  string line2 =
      "\033[94m┃\033[0m \033[94m┃┣━┫┃┃\033[34m┃┃┃┃\033[0m \033[34m┃┃\033[0m";
  string line3 = "\033[34m┗━┛╹\033[0m \033[34m╹╹\033[0m \033[34m╹╹\033[0m "
                 "\033[34m╹╺┻┛\033[0m";
  log<System::MESSAGE>("%s", line1.c_str());
  log<System::MESSAGE>("%s Version: %s", line2.c_str(), UAMMD_VERSION);
  log<System::MESSAGE>("%s", line3.c_str());
  log<System::MESSAGE>("Compiled at: %s %s", __DATE__, __TIME__);
#ifdef DOUBLE_PRECISION
  log<System::MESSAGE>("Compiled in double precision mode");
#else
  log<System::MESSAGE>("Compiled in single precision mode");
#endif
  std::time_t time =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  log<System::MESSAGE>("Computation started at %s", std::ctime(&time));
}

void System::printFarewell() {
  cudaDeviceSynchronize();
  std::time_t time =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  log<System::MESSAGE>("Computation finished at %s", std::ctime(&time));
  log<System::MESSAGE>("Time elapsed since creation: %fs", tim.toc());
  std::string line;
  fori(0, 60) line += "━";
  line += "┛";
  log<System::MESSAGE>("%s", line.c_str());
}

} // namespace uammd
#endif
