/*Raul P. Pelaez 2017-2021.
  Some utilities to create initial configurations of particles.


*/
#ifndef INITIALCONDITIONS_CUH
#define INITIALCONDITIONS_CUH

#include "global/defines.h"
#include "third_party/bravais/bravais.h"
#include "utils/vector.cuh"
#include <vector>
namespace uammd {
// Given a box size L, a number of particles N and a lattice type this function
// will return a vector of real4 positions placed in said lattice. Available
// lattices: sc, bcc, fcc, dia, hcp, sq, tri;
std::vector<real4> initLattice(real3 L, uint N, BRAVAISLAT lat) {
  std::vector<float4> pos(N, make_float4(0));
  Bravais((float *)pos.data(), lat, /*lattice type*/
          N, L.x, L.y, L.z, 0.0f,   /*Color*/
          NULL, NULL,               /*Basis and vector files*/
          false);                   /*Keep aspect ratio*/
  for (uint i = 0u; i < N; i++) {
    pos[i] += make_float4(0.56f, 0.56f, 0.56f, 0.0f);
    if (L.z == real(0.0))
      pos[i].z = 0.0f;
  }
  std::vector<real4> pos_real(N);
  for (uint i = 0u; i < N; i++)
    pos_real[i] = make_real4(pos[i].x, pos[i].y, pos[i].z, real(0.0));
  return pos_real;
}
} // namespace uammd
#endif
