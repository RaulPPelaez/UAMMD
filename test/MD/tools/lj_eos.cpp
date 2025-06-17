//*********************************************************************************************************************
//* Fundamental equation of state correlation for the Lennard-Jones truncated
//and shifted (rc=2.5*sigma) model fluid  *
//*                               M. Thol, G. Rutkai, R. Span, J. Vrabec, R.
//Lustig                                   *
//* *
//*   Email: jadran.vrabec@upb.de; Address: Thet, Universtität Paderborn,
//Warburger Str. 100, Paderborn, Germany      *
//* *
//*********************************************************************************************************************

#include <iomanip>
#include <iostream>
#include <math.h>
using namespace std;

// Derivatives of the polynomial terms up to arbitrary order:
double PolyTerm(double tau, double delta, double n, double t, double d,
                int ordertau, int orderdelta) {
  double product_tau = 1.0, product_delta = 1.0;

  for (int i = 0; i < ordertau; i++)
    product_tau = product_tau * (t - i);
  for (int i = 0; i < orderdelta; i++)
    product_delta = product_delta * (d - i);

  return n * product_tau * pow(tau, t - ordertau) * product_delta *
         pow(delta, d - orderdelta);
}
// Derivatives of the exponential terms up to order "ordertau"+"orderdelta"=2,
// where "ordertau" and "orderdelta" stand for the order of derivation with
// respect to tau and delta, respectively:
double ExpTerm(double tau, double delta, double n, double t, double d, double l,
               int ordertau, int orderdelta) {
  double product_tau = 1.0, g = 1.0;

  for (int i = 0; i < ordertau; i++)
    product_tau = product_tau * (t - i);

  if (orderdelta == 0)
    return n * product_tau * pow(tau, t - ordertau) * exp(-g * pow(delta, l)) *
           pow(delta, d);
  if (orderdelta == 1)
    return n * product_tau * pow(tau, t - ordertau) * exp(-g * pow(delta, l)) *
           (pow(delta, (d - 1.0)) * d - pow(delta, (d + l - 1.0)) * g * l);
  if (orderdelta == 2)
    return n * product_tau * pow(tau, t - ordertau) * exp(-g * pow(delta, l)) *
           (pow(delta, (d - 2.0)) * pow(d, 2.0) - pow(delta, (d - 2.0)) * d -
            2.0 * pow(delta, (d - 2.0 + l)) * d * g * l -
            pow(delta, (d - 2.0 + l)) * g * pow(l, 2.0) +
            pow(delta, (d - 2.0 + l)) * g * l +
            pow(delta, (d + 2.0 * l - 2.0)) * pow(g, 2.0) * pow(l, 2.0));
}
// Derivatives of the Gaussian terms up to order "ordertau"+"orderdelta"=2,
// where "ordertau" and "orderdelta" stand for the order of derivation with
// respect to tau and delta, respectively:
double GaussTerm(double tau, double delta, double n, double t, double d,
                 double eta, double beta, double gamma, double epsilon,
                 int ordertau, int orderdelta) {
  if (ordertau == 0 && orderdelta == 0)
    return n * pow(tau, t) * pow(delta, d) *
           exp(-eta * pow((delta - epsilon), 2.0) -
               beta * pow((tau - gamma), 2.0));
  if (ordertau == 1 && orderdelta == 0)
    return n * pow(delta, d) *
           exp(-eta * pow(delta, 2.0) + 2.0 * eta * delta * epsilon -
               eta * pow(epsilon, 2.0) - beta * pow(tau, 2.0) +
               2.0 * beta * tau * gamma - beta * pow(gamma, 2.0)) *
           (pow(tau, t - 1.0) * t - 2.0 * pow(tau, t + 1.0) * beta +
            2.0 * pow(tau, t) * beta * gamma);
  if (ordertau == 0 && orderdelta == 1)
    return n * pow(tau, t) *
           exp(-eta * pow(delta, 2.0) + 2.0 * eta * delta * epsilon -
               eta * pow(epsilon, 2.0) - beta * pow(tau, 2.0) +
               2.0 * beta * tau * gamma - beta * pow(gamma, 2.0)) *
           (pow(delta, d - 1.0) * d - 2.0 * pow(delta, d + 1.0) * eta +
            2.0 * pow(delta, d) * eta * epsilon);
  if (ordertau == 2 && orderdelta == 0)
    return n * pow(delta, d) *
           exp(-eta * pow(delta, 2.0) + 2.0 * eta * delta * epsilon -
               eta * pow(epsilon, 2.0) - beta * pow(tau, 2.0) +
               2.0 * beta * tau * gamma - beta * pow(gamma, 2.0)) *
           (pow(tau, t - 2.0) * pow(t, 2.0) - pow(tau, t - 2.0) * t -
            4.0 * pow(tau, t) * t * beta +
            4.0 * pow(tau, t - 1.0) * t * beta * gamma -
            2.0 * pow(tau, t) * beta +
            4.0 * pow(tau, t + 2.0) * pow(beta, 2.0) -
            8.0 * pow(tau, t + 1.0) * pow(beta, 2.0) * gamma +
            4.0 * pow(tau, t) * pow(beta, 2.0) * pow(gamma, 2.0));
  if (ordertau == 1 && orderdelta == 1)
    return n *
           exp(-eta * pow(delta, 2.0) + 2.0 * eta * delta * epsilon -
               eta * pow(epsilon, 2.0) - beta * pow(tau, 2.0) +
               2.0 * beta * tau * gamma - beta * pow(gamma, 2.0)) *
           (pow(tau, t - 1.0) * t * pow(delta, d - 1.0) * d -
            2.0 * pow(tau, t - 1.0) * t * pow(delta, d + 1.0) * eta +
            2.0 * pow(tau, t - 1.0) * t * pow(delta, d) * eta * epsilon -
            2.0 * pow(tau, t + 1.0) * pow(delta, d - 1.0) * d * beta +
            2.0 * pow(tau, t) * pow(delta, d - 1.0) * d * beta * gamma +
            4.0 * pow(tau, t + 1.0) * pow(delta, d + 1.0) * beta * eta -
            4.0 * pow(tau, t + 1.0) * pow(delta, d) * beta * eta * epsilon -
            4.0 * pow(tau, t) * pow(delta, d + 1.0) * beta * eta * gamma +
            4.0 * pow(tau, t) * pow(delta, d) * beta * eta * gamma * epsilon);
  if (ordertau == 0 && orderdelta == 2)
    return n * pow(tau, t) *
           exp(-eta * pow(delta, 2.0) + 2.0 * eta * delta * epsilon -
               eta * pow(epsilon, 2.0) - beta * pow(tau, 2.0) +
               2.0 * beta * tau * gamma - beta * pow(gamma, 2.0)) *
           (pow(delta, d - 2.0) * pow(d, 2.0) - pow(delta, d - 2.0) * d -
            4.0 * pow(delta, d) * d * eta +
            4.0 * pow(delta, d - 1.0) * d * eta * epsilon -
            2.0 * pow(delta, d) * eta +
            4.0 * pow(delta, d + 2.0) * pow(eta, 2.0) -
            8.0 * pow(delta, d + 1.0) * pow(eta, 2.0) * epsilon +
            4.0 * pow(delta, d) * pow(eta, 2.0) * pow(epsilon, 2.0));
}

double ReturnTermValue(int i, double tau, double delta, int ordertau,
                       int orderdelta, double **BasisFunPar) {
  if (fabs(BasisFunPar[i][0] - 1.0) < 0.01)
    return PolyTerm(tau, delta, BasisFunPar[i][1], BasisFunPar[i][2],
                    BasisFunPar[i][3], ordertau, orderdelta);
  if (fabs(BasisFunPar[i][0] - 2.0) < 0.01)
    return ExpTerm(tau, delta, BasisFunPar[i][1], BasisFunPar[i][2],
                   BasisFunPar[i][3], BasisFunPar[i][4], ordertau, orderdelta);
  if (fabs(BasisFunPar[i][0] - 3.0) < 0.01)
    return GaussTerm(tau, delta, BasisFunPar[i][1], BasisFunPar[i][2],
                     BasisFunPar[i][3], BasisFunPar[i][5], BasisFunPar[i][6],
                     BasisFunPar[i][7], BasisFunPar[i][8], ordertau,
                     orderdelta);
}

int main(int argc, char *argv[]) {
  // Critical temperature and density:
  double Tc = 1.086, Rhoc = 0.319;

  // Matrix that contains the parameters of the FEOS correlation
  double **BasisFunPar;
  int NTerms = 21, NParameters = 9;

  BasisFunPar = new double *[NTerms];
  for (int i = 0; i < NTerms; i++)
    BasisFunPar[i] = new double[NParameters];

  // TermID, n, t, d, l, eta, beta, gamma, epsilon
  BasisFunPar[0][0] = 1.0;
  BasisFunPar[0][1] = 0.0156060840;
  BasisFunPar[0][2] = 1.000;
  BasisFunPar[0][3] = 4.0;
  BasisFunPar[0][4] = 0.0;
  BasisFunPar[0][5] = 0.00;
  BasisFunPar[0][6] = 0.00;
  BasisFunPar[0][7] = 0.00;
  BasisFunPar[0][8] = 0.00;
  BasisFunPar[1][0] = 1.0;
  BasisFunPar[1][1] = 1.7917527000;
  BasisFunPar[1][2] = 0.304;
  BasisFunPar[1][3] = 1.0;
  BasisFunPar[1][4] = 0.0;
  BasisFunPar[1][5] = 0.00;
  BasisFunPar[1][6] = 0.00;
  BasisFunPar[1][7] = 0.00;
  BasisFunPar[1][8] = 0.00;
  BasisFunPar[2][0] = 1.0;
  BasisFunPar[2][1] = -1.9613228000;
  BasisFunPar[2][2] = 0.583;
  BasisFunPar[2][3] = 1.0;
  BasisFunPar[2][4] = 0.0;
  BasisFunPar[2][5] = 0.00;
  BasisFunPar[2][6] = 0.00;
  BasisFunPar[2][7] = 0.00;
  BasisFunPar[2][8] = 0.00;
  BasisFunPar[3][0] = 1.0;
  BasisFunPar[3][1] = 1.3045604000;
  BasisFunPar[3][2] = 0.662;
  BasisFunPar[3][3] = 2.0;
  BasisFunPar[3][4] = 0.0;
  BasisFunPar[3][5] = 0.00;
  BasisFunPar[3][6] = 0.00;
  BasisFunPar[3][7] = 0.00;
  BasisFunPar[3][8] = 0.00;
  BasisFunPar[4][0] = 1.0;
  BasisFunPar[4][1] = -1.8117673000;
  BasisFunPar[4][2] = 0.870;
  BasisFunPar[4][3] = 2.0;
  BasisFunPar[4][4] = 0.0;
  BasisFunPar[4][5] = 0.00;
  BasisFunPar[4][6] = 0.00;
  BasisFunPar[4][7] = 0.00;
  BasisFunPar[4][8] = 0.00;
  BasisFunPar[5][0] = 1.0;
  BasisFunPar[5][1] = 0.1548399700;
  BasisFunPar[5][2] = 0.870;
  BasisFunPar[5][3] = 3.0;
  BasisFunPar[5][4] = 0.0;
  BasisFunPar[5][5] = 0.00;
  BasisFunPar[5][6] = 0.00;
  BasisFunPar[5][7] = 0.00;
  BasisFunPar[5][8] = 0.00;
  BasisFunPar[6][0] = 2.0;
  BasisFunPar[6][1] = -0.4751098200;
  BasisFunPar[6][2] = 1.286;
  BasisFunPar[6][3] = 1.0;
  BasisFunPar[6][4] = 1.0;
  BasisFunPar[6][5] = 0.00;
  BasisFunPar[6][6] = 0.00;
  BasisFunPar[6][7] = 0.00;
  BasisFunPar[6][8] = 0.00;
  BasisFunPar[7][0] = 2.0;
  BasisFunPar[7][1] = -0.5842280700;
  BasisFunPar[7][2] = 1.960;
  BasisFunPar[7][3] = 1.0;
  BasisFunPar[7][4] = 2.0;
  BasisFunPar[7][5] = 0.00;
  BasisFunPar[7][6] = 0.00;
  BasisFunPar[7][7] = 0.00;
  BasisFunPar[7][8] = 0.00;
  BasisFunPar[8][0] = 2.0;
  BasisFunPar[8][1] = -0.5060736400;
  BasisFunPar[8][2] = 2.400;
  BasisFunPar[8][3] = 3.0;
  BasisFunPar[8][4] = 2.0;
  BasisFunPar[8][5] = 0.00;
  BasisFunPar[8][6] = 0.00;
  BasisFunPar[8][7] = 0.00;
  BasisFunPar[8][8] = 0.00;
  BasisFunPar[9][0] = 2.0;
  BasisFunPar[9][1] = 0.1163964400;
  BasisFunPar[9][2] = 1.700;
  BasisFunPar[9][3] = 2.0;
  BasisFunPar[9][4] = 1.0;
  BasisFunPar[9][5] = 0.00;
  BasisFunPar[9][6] = 0.00;
  BasisFunPar[9][7] = 0.00;
  BasisFunPar[9][8] = 0.00;
  BasisFunPar[10][0] = 2.0;
  BasisFunPar[10][1] = -0.2009241200;
  BasisFunPar[10][2] = 3.000;
  BasisFunPar[10][3] = 2.0;
  BasisFunPar[10][4] = 2.0;
  BasisFunPar[10][5] = 0.00;
  BasisFunPar[10][6] = 0.00;
  BasisFunPar[10][7] = 0.00;
  BasisFunPar[10][8] = 0.00;
  BasisFunPar[11][0] = 2.0;
  BasisFunPar[11][1] = -0.0948852040;
  BasisFunPar[11][2] = 1.250;
  BasisFunPar[11][3] = 5.0;
  BasisFunPar[11][4] = 1.0;
  BasisFunPar[11][5] = 0.00;
  BasisFunPar[11][6] = 0.00;
  BasisFunPar[11][7] = 0.00;
  BasisFunPar[11][8] = 0.00;
  BasisFunPar[12][0] = 3.0;
  BasisFunPar[12][1] = 0.0094333106;
  BasisFunPar[12][2] = 3.600;
  BasisFunPar[12][3] = 1.0;
  BasisFunPar[12][4] = 0.0;
  BasisFunPar[12][5] = 4.70;
  BasisFunPar[12][6] = 20;
  BasisFunPar[12][7] = 1.0;
  BasisFunPar[12][8] = 0.55;
  BasisFunPar[13][0] = 3.0;
  BasisFunPar[13][1] = 0.3044462800;
  BasisFunPar[13][2] = 2.080;
  BasisFunPar[13][3] = 1.0;
  BasisFunPar[13][4] = 0.0;
  BasisFunPar[13][5] = 1.92;
  BasisFunPar[13][6] = 0.77;
  BasisFunPar[13][7] = 0.5;
  BasisFunPar[13][8] = 0.7;
  BasisFunPar[14][0] = 3.0;
  BasisFunPar[14][1] = -0.0010820946;
  BasisFunPar[14][2] = 5.240;
  BasisFunPar[14][3] = 2.0;
  BasisFunPar[14][4] = 0.0;
  BasisFunPar[14][5] = 2.70;
  BasisFunPar[14][6] = 0.5;
  BasisFunPar[14][7] = 0.8;
  BasisFunPar[14][8] = 2.0;
  BasisFunPar[15][0] = 3.0;
  BasisFunPar[15][1] = -0.0996933910;
  BasisFunPar[15][2] = 0.960;
  BasisFunPar[15][3] = 3.0;
  BasisFunPar[15][4] = 0.0;
  BasisFunPar[15][5] = 1.49;
  BasisFunPar[15][6] = 0.8;
  BasisFunPar[15][7] = 1.5;
  BasisFunPar[15][8] = 1.14;
  BasisFunPar[16][0] = 3.0;
  BasisFunPar[16][1] = 0.0091193522;
  BasisFunPar[16][2] = 1.360;
  BasisFunPar[16][3] = 3.0;
  BasisFunPar[16][4] = 0.0;
  BasisFunPar[16][5] = 0.65;
  BasisFunPar[16][6] = 0.4;
  BasisFunPar[16][7] = 0.7;
  BasisFunPar[16][8] = 1.2;
  BasisFunPar[17][0] = 3.0;
  BasisFunPar[17][1] = 0.1297054300;
  BasisFunPar[17][2] = 1.655;
  BasisFunPar[17][3] = 2.0;
  BasisFunPar[17][4] = 0.0;
  BasisFunPar[17][5] = 1.73;
  BasisFunPar[17][6] = 0.43;
  BasisFunPar[17][7] = 1.6;
  BasisFunPar[17][8] = 1.31;
  BasisFunPar[18][0] = 3.0;
  BasisFunPar[18][1] = 0.0230360300;
  BasisFunPar[18][2] = 0.900;
  BasisFunPar[18][3] = 1.0;
  BasisFunPar[18][4] = 0.0;
  BasisFunPar[18][5] = 3.70;
  BasisFunPar[18][6] = 8.0;
  BasisFunPar[18][7] = 1.3;
  BasisFunPar[18][8] = 1.14;
  BasisFunPar[19][0] = 3.0;
  BasisFunPar[19][1] = -0.0826710730;
  BasisFunPar[19][2] = 0.860;
  BasisFunPar[19][3] = 2.0;
  BasisFunPar[19][4] = 0.0;
  BasisFunPar[19][5] = 1.90;
  BasisFunPar[19][6] = 3.3;
  BasisFunPar[19][7] = 0.6;
  BasisFunPar[19][8] = 0.53;
  BasisFunPar[20][0] = 3.0;
  BasisFunPar[20][1] = -2.2497821000;
  BasisFunPar[20][2] = 3.950;
  BasisFunPar[20][3] = 3.0;
  BasisFunPar[20][4] = 0.0;
  BasisFunPar[20][5] = 13.20;
  BasisFunPar[20][6] = 114.0;
  BasisFunPar[20][7] = 1.3;
  BasisFunPar[20][8] = 0.96;

  double T, Rho, tau, delta;

  cout << endl
       << "Specify a state point located in the homogeneous fluid region (gas "
          "or liquid) in Lennard-Jones reduced units."
       << endl;
  cout << endl
       << "(Caution: Formally, the program will evaluate state points and does "
          "not give warning if they are located outside the target "
          "(homogeneous) region.)"
       << endl;
  cout << endl << "Temperature: ";
  cin >> T;
  cout << endl << "Density:     ";
  cin >> Rho;

  tau = Tc / T;
  delta = Rho / Rhoc;

  // residual and dimensionless Helmholtz free energy derivatives
  double A00 = 0.0, A10 = 0.0, A01 = 0.0, A20 = 0.0, A11 = 0.0, A02 = 0.0;

  for (int i = 0; i < NTerms; i++) {
    A00 += ReturnTermValue(i, tau, delta, 0, 0, BasisFunPar);
    A10 += ReturnTermValue(i, tau, delta, 1, 0, BasisFunPar) * tau;
    A01 += ReturnTermValue(i, tau, delta, 0, 1, BasisFunPar) * delta;
    A20 += ReturnTermValue(i, tau, delta, 2, 0, BasisFunPar) * tau * tau;
    A11 += ReturnTermValue(i, tau, delta, 1, 1, BasisFunPar) * tau * delta;
    A02 += ReturnTermValue(i, tau, delta, 0, 2, BasisFunPar) * delta * delta;
  }

  double p, ures, hres, cvres, cpres, mures, dpdrho, dpdt;

  // Corrected:
  // p=(1.0+A01)/T/Rho -> p=(1.0+A01)*T*Rho
  // ures = A10 / T -> ures=A10*T;;
  p = (1.0 + A01) * T * Rho;
  mures = A00 + A01;
  ures = A10 * T;
  hres = (1.0 + A01 + A10) * T;
  cvres = -A20;
  cpres = -A20 +
          (1.0 + A01 - A11) * (1.0 + A01 - A11) / (1.0 + 2.0 * A01 + A02) - 1.0;
  dpdrho = T * (1.0 + 2.0 * A01 + A02);
  dpdt = Rho * (1.0 + A01 - A11);

  cout << endl << "Properties in Lennard-Jones reduced units:" << endl;

  cout << setprecision(15) << endl
       << "Pressure:                " << p << " (total)";
  cout << endl << "Chemical Potential:      " << mures << " (residual)";
  cout << endl << "Internal Energy:         " << ures << " (residual)";
  cout << endl << "Enthalpy:                " << hres << " (residual)";
  cout << endl << "Isochoric Heat Capacity: " << cvres << " (residual)";
  cout << endl << "Isobaric  Heat Capacity: " << cpres << " (residual)";
  cout << endl << "dp/d(rho)  at T=const.:  " << dpdrho << " (total)";
  cout << endl << "dp/dT  at rho=const.:    " << dpdt << " (total)" << endl;

  cout << endl
       << "Helmholtz free energy derivatives (residual and dimensionless):"
       << endl;

  cout << endl << "A00: " << A00;
  cout << endl << "A10: " << A10;
  cout << endl << "A01: " << A01;
  cout << endl << "A20: " << A20;
  cout << endl << "A11: " << A11;
  cout << endl << "A02: " << A02 << endl;

  cout << endl
       << "The mixed partial derivative "
          "Axy=d^(x+y)(F/(NkBT))/d(1/T)^x/d(rho)^y * (1/T)^x * (rho)^y, where "
          "F is the Helmholtz free energy [J] (extensive), N is the number of "
          "particles, kB is the Boltzmann constant [J/K], T is the temperature "
          "[K], and rho is the density [mol/m3].";
  cout << endl
       << "Note that Axy is dimensionless, independent from the choice of "
          "units (Lennard-Jones or SI)."
       << endl;

  for (int i = 0; i < NTerms; i++)
    delete[] BasisFunPar[i];
  delete[] BasisFunPar;

  return 1;
}
