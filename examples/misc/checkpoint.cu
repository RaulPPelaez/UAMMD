/*Raul P. Pelaez 2021

  Example on how to save and restore the particle properties to a file.

 */
#include "utils/checkpoint.h"
using namespace uammd;

// Creates and fills a ParticleData instance with some arbitrary data. In this
// case populating position and charges of each particle
auto createParticleData(std::shared_ptr<System> sys) {
  constexpr int N = 10;
  auto pd = std::make_shared<ParticleData>(sys, N);
  auto pos = pd->getPos(access::cpu, access::write);
  std::fill(pos.begin(), pos.end(), real4() + 1);
  auto charge = pd->getCharge(access::cpu, access::write);
  std::fill(charge.begin(), charge.end(), 2);
  return pd;
}

int main() {
  // Create a System
  auto sys = std::make_shared<System>();
  { // Create and fill a ParticleData and save it to a file
    auto pd = createParticleData(sys);
    saveParticleData("pd.dat", pd);
    // Now the ParticleData instance is destroyed
  }
  // Restore the saved ParticleData from the file
  auto pd = restoreParticleData("pd.dat", sys);
  return 0;
}
