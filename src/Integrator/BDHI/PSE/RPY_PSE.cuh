/*Raul P. Pelaez 2017. Contains the near, real space, part of the Positively
Split Edwald Rotne Prager Yamakawa tensor.

Calling the operator () with a distance, it will return two values, F and G as
in:

Mobility = Mreal + Mwave
  Mreal(r) = F(r)(I-r^r) + G(r)(r^r)


References:

[1]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations
           -  https://arxiv.org/pdf/1611.09322.pdf
[2]  Spectral accuracy in fast Ewald-based methods for particle simulations
           -  http://www.sciencedirect.com/science/article/pii/S0021999111005092

 */
#ifndef BDHI_PSE_RPY_PSE_CUH
#define BDHI_PSE_RPY_PSE_CUH
#include"global/defines.h"

namespace uammd{
  namespace BDHI{
    struct RPYPSE_near{
    private:
      const double rh, psi;
      const double normalization; //Divide F and G by this.
      const double rcut;

    public:
      RPYPSE_near(real rh, real psi, real normalization, real rcut):
	rh(rh), psi(psi), normalization(normalization), rcut(rcut){}

      double2 FandG(double r) const;
      inline real2 operator()(double r) const{
	return make_real2(this->FandG(r)/normalization);
      }

    private:

      double params2FG(double r,
		       double f0, double f1, double f2, double f3,
		       double f4, double f5, double f6, double f7) const;
    };


    double2 RPYPSE_near::FandG(double r) const{
      double r2 = r*r;
      if(r>=rcut) return make_double2(0.0, 0.0);
      if(r<=real(0.0)){
      /*(6*pi*vis*a)*Mr(0) = F(0)(I-r^r) + G(0)(r^r) = F(0) = (6*pi*vis*a)*Mii_r .
	See eq. A4 in [1] and RPYPSE_nearTextures*/
	double pi = M_PI;
	double f0 = (1.0/(4*sqrt(pi)*psi*rh))*(1-exp(-4*rh*rh*psi*psi)+4*sqrt(pi)*rh*psi*std::erfc(2*rh*psi));
	return make_double2(f0, 0);
      }
      double a2mr = 2*rh-r;
      double a2pr = 2*rh+r;
      double rh2 = rh*rh;
      double rh4 = rh2*rh2;
      double psi2 = psi*psi;
      double psi3 = psi2*psi;
      double psi4 = psi2*psi2;
      double r3 = r2*r;
      double r4 = r3*r;
      double f0, f1, f2, f3, f4 ,f5, f6, f7;
      double g0, g1, g2, g3, g4 ,g5, g6, g7;
      if(r>2*rh){
	f0 =(64.0*rh4*psi4 + 96.0*rh2*r2*psi4
	     - 128.0*rh*r3*psi4 + 36.0*r4*psi4-3.0)/
	  (128.0*rh*r3*psi4);
	f4 = (3.0-4.0*psi4*a2mr*a2mr*(4.0*rh2+4.0*rh*r+9.0*r2))/
	  (256.0*rh*r3*psi4);
	f5 = 0;
	g0 = (-64.0*rh4*psi4+96.0*rh2*r2*psi4-64.0*rh*r3*psi4 + 12.0*r4*psi4 +3.0)/
	  (64.0*rh*r3*psi4);
	g4 = (4.0*psi4*a2mr*a2mr*a2mr*(2.0*rh+3.0*r)-3.0)/(128.0*rh*r3*psi4);
	g5 = 0;
      }
      else{
	f0 = (-16.0*rh4-24.0*rh2*r2+32.0*rh*r3-9.0*r4)/
	  (32.0*rh*r3);
	f4 = 0;
	f5 = (4.0*psi4*a2mr*a2mr*(4.0*rh2+4.0*rh*r+9.0*r2)-3.0)/
	  (256.0*rh*r3*psi4);
	g0 = a2mr*a2mr*a2mr*(2.0*rh+3.0*r)/(16.0*rh*r3);
	g4 = 0;
	g5 = (3.0 - 4.0*psi4*a2mr*a2mr*a2mr*(2.0*rh+3.0*r))/(128.0*rh*r3*psi4);
      }
      f1 = (-2.0*psi2*a2pr*(4.0*rh2-4.0*rh*r+9.0*r2) + 2.0*rh -3.0*r)/
	(128.0*rh*r3*psi3*sqrt(M_PI));
      f2 = (2.0*psi2*a2mr*(4.0*rh2+4.0*rh*r+9.0*r2)-2.0*rh-3.0*r)/
	(128.0*rh*r3*psi3*sqrt(M_PI));
      f3 = 3.0*(6.0*r2*psi2+1.0)/(64.0*sqrt(M_PI)*rh*r2*psi3);
      f6 = (4.0*psi4*a2pr*a2pr*(4.0*rh2-4.0*rh*r+9.0*r2)-3.0)/
	(256.0*rh*r3*psi4);
      f7 = 3.0*(1.0-12.0*r4*psi4)/(128.0*rh*r3*psi4);
      g1 = (2.0*psi2*a2pr*a2pr*(2.0*rh-3.0*r)-2.0*rh+3.0*r)/
	(64.0*sqrt(M_PI)*rh*r3*psi3);
      g2 = (-2.0*psi2*a2mr*a2mr*(2.0*rh+3.0*r)+2.0*rh+3.0*r)/
	(64.0*sqrt(M_PI)*rh*r3*psi3);
      g3 = (3.0*(2.0*r2*psi2-1.0))/(32.0*sqrt(M_PI)*rh*r2*psi3);
      g6 = (3.0-4.0*psi4*(2.0*rh-3.0*r)*a2pr*a2pr*a2pr)/(128.0*rh*r3*psi4);
      g7 = -3.0*(4.0*r4*psi4+1.0)/(64.0*rh*r3*psi4);
      return {params2FG(r, f0, f1, f2, f3, f4, f5, f6, f7), params2FG(r, g0, g1, g2, g3, g4, g5, g6, g7)};
    }



    double RPYPSE_near::params2FG(double r,
				  double f0, double f1, double f2, double f3,
				  double f4, double f5, double f6, double f7) const{
      double psisq = psi*psi;
      double a2mr = 2*rh-r;
      double a2pr = 2*rh+r;
      double rsq = r*r;
      return  f0 + f1*exp(-psisq*a2pr*a2pr) +
	f2*exp(-a2mr*a2mr*psisq) + f3*exp(-psisq*rsq)+
	f4*erfc(a2mr*psi) + f5*erfc(-a2mr*psi)+
	f6*erfc(a2pr*psi) + f7*erfc(r*psi);
    }


  }
}
#endif
