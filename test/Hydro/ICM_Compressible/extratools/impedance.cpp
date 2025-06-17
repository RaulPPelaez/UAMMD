/*Raul P.Pelaez 2022. Computes the impedance of a fluid at the wall when the
   wall is moving with an oscillatory velocity.

 */
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
using real = double;
using complex = std::complex<real>;

auto impedance(real delta, real L) {
  const complex alpha = (1.0 - complex{0, 1}) / delta;
  return -(1.0 - complex{0, 1}) * cosh(alpha * L) / sinh(alpha * L);
}

int main(int argc, char *argv[]) {
  real viscosity = std::atof(argv[1]);
  real L = std::atof(argv[2]);
  real vamplitude = std::atof(argv[3]);
  real time = std::atof(argv[4]);
  real rho0 = 1;
  real w = std::atof(argv[5]);
  real delta = sqrt(2 * viscosity / (rho0 * w));
  auto Z = impedance(delta, L);
  std::cout << std::real(Z) << " " << std::imag(Z) << std::endl;
  return 0;
}
