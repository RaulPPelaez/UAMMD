/*Raul P. Pelaez 2019. Error handling example.
  Shows how to deal with exceptions when something goes wrong in UAMMD.
*/

#include <ParticleData/ParticleData.cuh>
#include <cstdlib>
#include <iostream>
#include <uammd.cuh>
#include <utils/exception.h>

using uammd::access;
using uammd::ParticleData;
using uammd::System;

std::shared_ptr<System> createSystem(int argc, char **argv) {
  try {
    return std::make_shared<System>(argc, argv);
  } catch (uammd::exception &e) {
    System::log<System::WARNING>("Recovering from System exception");
    System::log<System::ERROR>("Backtrace:");
    uammd::backtrace_nested_exception(e);
    return createSystem(0, nullptr);
  }
}

// Try to call this program with an invalid input, for example ./a.out --device
// 1000 System creation will fail and try to recover
int main(int argc, char **argv) {

  auto sys = createSystem(argc, argv);
  int N = 100;
  auto pd = std::make_shared<ParticleData>(N, sys);
  auto pos = pd->getPos(access::location::cpu, access::mode::write);

  try {
    auto pos_illegal = pd->getPos(access::location::cpu, access::mode::read);
  } catch (uammd::illegal_property_access &e) {
    sys->log<System::WARNING>("Recovering from access exception");
    uammd::backtrace_nested_exception(e);
  }
  sys->finish();
  return EXIT_SUCCESS;
};