/* Raul P. Pelaez 2022. Prints the velocity of a fluid between two walls when
   one of the walls is moving with a velocity vx(z=0)=v0*cos(wt).

   Rafa gave me the theoretical expression.
 */
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
using real = double;
using complex = std::complex<real>;

auto vzhat(real z, real delta, real L) {
  const complex alpha = (1.0 - complex{0, 1}) / delta;
  return sinh(alpha * (L - z)) / sinh(alpha * L);
}

auto vz_theory(real z, real vamplitude, real w, real t, real delta, real L) {
  const auto vzh = vzhat(z, delta, L);
  const real vz =
      vamplitude * (std::real(vzh) * cos(w * t) + std::imag(vzh) * sin(w * t));
  return vz;
}

int main(int argc, char *argv[]) {
  real viscosity = std::atof(argv[1]);
  real L = std::atof(argv[2]);
  real vamplitude = std::atof(argv[3]);
  real time = std::atof(argv[4]);
  real rho0 = 1;
  real frequency = std::atof(argv[5]);
  real w = 2 * M_PI * frequency;
  real delta = sqrt(2 * viscosity / (rho0 * w));
  real z = std::atof(argv[6]);
  real vx_z = vz_theory(z, vamplitude, w, time, delta, L);
  std::cout << vx_z << std::endl;
  return 0;
}
