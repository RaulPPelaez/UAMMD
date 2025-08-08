/*Raul P. Pelaez 2019. Checks the cross mobility of a group of particles with
  different sizes.

  Let us perform a Brownian Hydrodynamics simulation with N non interacting
  particles with two radius, a1 and a2, at zero temperature and with open
  boundaries. Let us put N-1 particles with radius a1 and the first particle
  (i=0) with radius a2 and pull particle i=0 with a constant force F.x = 1;

  In each step of the simulation the displacement of a certain particle, except
  the i=0, will be given by:

  dxi/dt = M*F = M0i = f(r)*I + g(r)/r^2 * (r\diadic r)

  This code computes f(r) and g(r) from the displacements of particles at each
  time step and compares it with the theoretical RPY tensor with particles of
  different sizes. This code prints: r abs((f-ftheo)/ftheo) abs((g-gtheo)/gtheo)

  For all particles and time steps it encounters. This means that if all goes
  well the first and second columns should always be zero, with deviations from
  zero coming from numerical accuracy (typically around 1e-10 if the simulation
  is carried out with double precision).

  A data.main file must be present with information about the simulation,
  mainly:

  N              10000
  radius_min     1
  radius_max     0.1
  viscosity      1
  dt	       1
  nsteps	       50
  printSteps     1


  The simulation results must be piped to this code and be in superpunto format
  (frames separated with a line containing a #):

  #
  x y z
  .
  .
  .
  #
  x y z
  .
  .
  .
  The first particle in the list should be the one that is being pulled.


  USAGE:

  ./bdhi | ./process | tee deviation_from_theory | awk
  '$1>maxf{maxf=$1}$2>maxg{maxg=$2}END{print maxf, maxg}'

  This will print the maximum deviation from theory, which should be <1e-8 if
  pos.dat has 14 digits of precision and the simulation was carried with double
  precision.


 */
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

std::string grep(std::string lookfor) {
  char res[1024];
  FILE *file = popen(
      ("cat data.main | grep '^" + lookfor + "\\s' | awk '{print $2}'").c_str(),
      "r");
  auto ptr = fgets(res, sizeof(res), file);
  pclose(file);
  return string(res);
}

long double viscosity;
struct double2 {
  long double x, y;
};
double2 RPY_differentSizes(double r, double ai, double aj) {

  long double M0 = 1.0L / (6 * M_PIl * viscosity);
  const long double asum = ai + aj;
  const long double asub = fabsl(ai - aj);
  double2 c12;
  if (r > asum) {
    const long double invr = 1.0L / r;
    const long double pref = M0 * 3.0L * 0.25L * invr;
    const long double denom = (ai * ai + aj * aj) / (3.0L * r * r);
    c12.x = pref * (1.0L + denom);
    c12.y = pref * (1.0L - 3.0L * denom) * invr * invr;
  } else if (r > asub) {
    const long double pref = M0 / (ai * aj * 32.0L * r * r * r);
    long double num = asub * asub + 3.0L * r * r;
    c12.x = pref * (16.0L * r * r * r * asum - num * num);
    num = asub * asub - r * r;
    c12.y = pref * (3.0L * num * num) / (r * r);

  } else {
    c12.x = M0 / (ai > aj ? ai : aj);
    c12.y = 0;
  }
  return c12;
}

int main() {

  int N = std::atoi(grep("N").c_str());
  int printSteps = std::atoi(grep("printSteps").c_str());

  int Nsteps = std::atoi(grep("nsteps").c_str());
  double dt = std::stod(grep("dt").c_str()) * printSteps;
  double radius_max = std::stod(grep("radius_max").c_str());
  double radius_min = std::stod(grep("radius_min").c_str());
  viscosity = std::stod(grep("viscosity").c_str());

  vector<long double> pos(3 * N, 0);
  vector<long double> posprev(3 * N, 0);
  string line;

  getline(cin, line);

  for (int i = 0; i < N; i++) {
    getline(cin, line);
    char *ptr = nullptr;
    posprev[3 * i] = std::strtod(line.c_str(), &ptr);
    posprev[3 * i + 1] = std::strtod(ptr, &ptr);
    posprev[3 * i + 2] = std::strtod(ptr, &ptr);
  }

  for (int j = 0; j < Nsteps - 1; j++) {
    getline(cin, line);
    for (int i = 0; i < N; i++) {
      getline(cin, line);
      char *ptr = nullptr;
      pos[3 * i] = std::strtod(line.c_str(), &ptr);
      pos[3 * i + 1] = std::strtod(ptr, &ptr);
      pos[3 * i + 2] = std::strtod(ptr, &ptr);
    }

    long double r0[3] = {posprev[0], posprev[1], posprev[2]};
    for (int i = 1; i < N; i++) {
      long double rij[3] = {r0[0] - posprev[3 * i], r0[1] - posprev[3 * i + 1],
                            r0[2] - posprev[3 * i + 2]};
      long double r = sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);

      long double vel[3] = {(pos[3 * i] - posprev[3 * i]) / dt,
                            (pos[3 * i + 1] - posprev[3 * i + 1]) / dt,
                            (pos[3 * i + 2] - posprev[3 * i + 2]) / dt};

      long double g = vel[1] * r * r / (rij[0] * rij[1]);
      long double f = vel[0] - g * (rij[0] * rij[0]) / (r * r);

      long double ftheo = RPY_differentSizes(r, radius_max, radius_min).x;
      long double ff = f - ftheo;
      long double gtheo =
          RPY_differentSizes(r, radius_max, radius_min).y * r * r;
      long double gg;
      if (gtheo < 1e-10 and g < 1e-10)
        gg = 0;
      else
        gg = g - gtheo;

      cout << setprecision(17) << r << " " << fabsl(ff / ftheo) << " "
           << fabsl(gg / (gtheo > 0 ? gtheo : 1)) << "\n";
      g = vel[2] * r * r / (rij[0] * rij[2]);
      f = vel[0] - g * (rij[0] * rij[0]) / (r * r);

      if (gtheo < 1e-10 and g < 1e-10)
        gg = 0;
      else
        gg = g - gtheo;
      ff = f - ftheo;
      cout << setprecision(17) << r << " " << fabsl(ff / ftheo) << " "
           << fabsl(gg / (gtheo > 0 ? gtheo : 1)) << "\n";
    }
    posprev = pos;
  }

  return 0;
}
