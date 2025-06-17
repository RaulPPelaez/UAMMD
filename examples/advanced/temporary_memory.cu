/* Raul P. Pelaez 2019. This example demonstrates the usage of the UAMMD
temporary device pool allocator You can use this allocator to request device
memory that will only be needed for a short time, or that must be
allocated/deallocated frequently (such as for a thrust algorithm). The requested
memory blocks are kept in a cache instead of being cudaFree'd, which makes
allocating/deallocating almost free when the same blocks are being constantly
allocated/deallocated.

System::allocator is a polymorphic allocator, which makes a
thrust::device_vector using it compatible with vectors usig the default
allocator.
 */
#include "uammd.cuh"
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <vector>

using namespace uammd;

int main() {
  auto sys = std::make_shared<System>();
  // Only the first iteration incurs a cudaMalloc, and cudaFree is called only
  // when sys goes out of scope.
  fori(0, 10) {
    thrust::device_vector<char, System::allocator_thrust<char>> vec;
    vec.resize(10000);
  }

  // You can interchange with a thrust vector using the default allocator.
  {
    thrust::device_vector<char, System::allocator_thrust<char>> vec;
    vec.resize(10000);
    // thrust::host_vector<char> host_copy_with_default_allocator(vec);
    thrust::device_vector<char> device_copy_with_default_allocator(vec);
  }
  {
    // Using the allcoator with a shared_ptr
    std::shared_ptr<int> temp;
    {
      auto alloc = sys->getTemporaryDeviceAllocator<int>();
      temp = std::shared_ptr<int>(alloc.allocate(1000),
                                  [=](int *ptr) { alloc.deallocate(ptr); });
    }
  }

  sys->finish();
  return 0;
}
