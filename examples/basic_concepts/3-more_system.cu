/* Raul P. Pelaez 2021
   More about System.
   Besides its logging capabilities, System also provides a random number
   generator that can be used to seed other generators throughout the code. We
   will also see other things that System can do.
 */

// uammd.cuh is the basic uammd include containing, among other things, the
// System struct.
#include <random>
#include <thrust/device_vector.h>
#include <uammd.cuh>
using namespace uammd;

int main(int argc, char *argv[]) {
  // Initialize System
  auto sys = std::make_shared<System>(argc, argv);
  // We can access the System's random number generator via sys->rng()
  // The first thing we should do is seed it with something controlled by us.
  // Since UAMMD modules will use this generator when in need of a seed, this
  // will allow us to have deterministic runs. Meaning that the same UAMMD code
  // ran with the same seed will result in the same random numbers. Besides
  // numerical error, coming from the non-sequential nature of GPU execution,
  // this gives reproducible runs. If not seeded, System will initialize the
  // generator using the current number of seconds since epoch.
  auto seed = 0x12345;
  // Alternatively, we could use C++'s random_device to seed differently each
  // run: std::random_device rd; auto seed = rd();
  sys->rng().setSeed(seed);
  // This rng can generate numbers in several ways, lets see a few:
  // A number chosen among all the representable by uint (0 and 2^32-1)
  uint integer = sys->rng().next32();
  sys->log<System::MESSAGE>("An integer number between 0 and 2^32-1: %u",
                            integer);
  // A number chosen among all the representable by uint64_t  (0 and 2^64-1)
  uint64_t long_integer = sys->rng().next();
  sys->log<System::MESSAGE>("An integer number between 0 and 2^64-1: %lu",
                            long_integer);
  // An uniform random number between 0 and1
  double uniform = sys->rng().uniform(0, 1);
  sys->log<System::MESSAGE>(
      "An uniformly distributed number between 0 and 1: %.13g", uniform);
  // A normally distributed number with 0 mean and standard deviation 1
  double normal = sys->rng().gaussian(0, 1);
  sys->log<System::MESSAGE>(
      "A normally distributed number with mean 0 and stdev 1: %.13g", normal);

  // System also allows to access the argc and argv passed at construction
  {
    auto argc = sys->getargc();
    auto argv = sys->getargv();
    sys->log<System::MESSAGE>(
        "The name of this executable is %s, %d arguments were passed to it.",
        argv[0], argc - 1);
  }

  // GPU memory allocation is really slow, for that matter System provides a C++
  // compatible GPU memory pool allocator.
  //  This allocator caches queries to it, so multiple allocations/deallocations
  //  of similar sizes will be almost instantaneous We will just mention it
  //  exists or now, though, and leave it to the advanced examples. auto alloc =
  //  sys->getTemporaryDeviceAllocator<double>(); thrust::device_vector<double,
  //  System::allocator_thrust<double>> vec(10000, alloc);

  // Destroy the UAMMD environment and exit
  sys->finish();
  return 0;
}
