
#include "utils.cuh"
#include<random>


real phi(real r, real rmax, real sigma){
  r= abs(r);
  real res = 0;
  if(r<=rmax)
    res=(1.0/(sigma*sqrt(2*M_PI)))*exp(-0.5*r*r/(sigma*sigma));
  return res;
}

void runSpreadingTest(real3 qi, real3 L, int3 n, int supp){
  real3 h = L/make_real3(n);
  real sigma = h.x;
  complex Fi = complex{1,0};
  bool error = false;
  int3 ci = chebyshev::doublyperiodic::Grid(Box(L), n).getCell(qi);
  System::log<System::MESSAGE>("Spreading a particle at %g %g %g (cell %d %d %d)",
			       qi.x, qi.y, qi.z,
			       ci.x, ci.y, ci.z);
  std::vector<complex> fr = spreadParticles({qi}, {Fi}, n, sigma, supp, L);
  complex maxErr{};
  int3 cellOfMaxErr={-1,-1,-1};
  Box box(L);
  box.setPeriodicity(1, 1, 0);
  for(int i = 0; i<n.x; i++){
    for(int j = 0; j<n.y; j++){
      for(int k = 0; k<n.z; k++){
	real3 r;
	r.x = -L.x*0.5 + (i+1e-10)*h.x-qi.x;
	r.y = -L.y*0.5 + (j+1e-10)*h.y-qi.y;
	r.z = 0.5*L.z*cospi(k/(n.z-1.0))-qi.z;
	r= box.apply_pbc(r);
	complex fijk = {0,0};
	real rmax = supp*h.x*0.5;
	fijk = Fi*phi(r.x, rmax, sigma)*phi(r.y, rmax, sigma)*phi(r.z, rmax, sigma);
 	int id = i + (j+k*n.y)*n.x;
	real errnorm = std::max(fijk.real(),fijk.imag());
	if(errnorm == 0){
	  errnorm = std::max(fr[id].real(),fr[id].imag());
	  if(errnorm==0)
	    errnorm = 1;
	}
	complex err = (fr[id] - fijk)/errnorm;
	err.real(abs(err.real()));
	err.imag(abs(err.imag()));
	if(maxErr.real() < err.real()){
	  maxErr.real(err.real());
	  maxErr.imag(err.imag());
	  cellOfMaxErr = {i,j,k};
	}
	if(norm(err)> 1e-13 or (norm(fr[id]) == 0 and norm(fijk) != 0) or
	   (norm(fijk) == 0 and norm(fr[id]) != 0)){
	  error = true;
	  System::log<System::ERROR>("Difference in cell %d %d %d: Found %g %g, expected %g %g (error %g)",
				     i,j,k, fr[id].real(), fr[id].imag(), fijk.real(), fijk.imag(),
				     thrust::norm(err));
	}
      }
    }
  }
  if(cellOfMaxErr.x >= 0 and norm(maxErr) > 1e-14){
    int id = cellOfMaxErr.x + (cellOfMaxErr.y+cellOfMaxErr.z*n.y)*n.x;
    real rmax = supp*h.x*0.5;
    real3 r;
    r.x = -L.x*0.5 + cellOfMaxErr.x*h.x-qi.x;
    r.y = -L.y*0.5 + cellOfMaxErr.y*h.y-qi.y;
    r.z = 0.5*L.z*cospi(cellOfMaxErr.z/(n.z-1.0))-qi.z;
    r= box.apply_pbc(r);
    complex fijk = {0,0};
    fijk = Fi*phi(r.x, rmax, sigma)*phi(r.y, rmax, sigma)*phi(r.z, rmax, sigma);
    System::log<System::MESSAGE>("Maximum error %g %g found at cell %d %d %d (expected %.15g %.15g, got %.15g %.15g)",
				 maxErr.real(), maxErr.imag(),
				 cellOfMaxErr.x, cellOfMaxErr.y, cellOfMaxErr.z,
				 fijk.real(), fijk.imag(),
				 fr[id].real(), fr[id].imag()
				 );
  }
  if(not error){
    System::log<System::MESSAGE>("[SUCCESS]");
  }
}

//Spreads particles forces and checks that the correct force density has been created.
void checkSpreading(){
  std::random_device r;
  std::default_random_engine e1(1234);
  std::uniform_real_distribution<real> uniform(-0.5, 0.5);
  real3 L = {1,1,1};
  int3 n = {16,16,16};
  //I check an even and an odd support
  int supp = 5;
  int ntest = 5;
  for(int test = 0; test<ntest; test++){
    real3 qi = make_real3(uniform(e1),uniform(e1), uniform(e1))*L;
    runSpreadingTest(qi, L,n, supp);
  }
  supp=6;
  for(int test = 0; test<ntest; test++){
    real3 qi = make_real3(uniform(e1),uniform(e1), uniform(e1))*L;
    runSpreadingTest(qi, L,n, supp);
  }
  runSpreadingTest(make_real3(0,0,-L.z*0.5), L,n, supp);
  runSpreadingTest(make_real3(0,0,L.z*0.5), L, n, supp);
  runSpreadingTest(make_real3(0,0,-L.z*0.5+0.001), L, n, supp);
  runSpreadingTest(make_real3(0,0,0.001), L,{128,128,128}, 128);
}

void runInterpolationTest(real3 qi, real3 L, int3 n, int supp){
  System::log<System::MESSAGE>("Interpolating particle at %g %g %g\n", qi.x, qi.y, qi.z);
  complex Fi = complex{1,0};
  real sigma = 0.1;
  std::vector<complex> fr = spreadParticles({qi}, {Fi}, n, sigma, supp, L);
  auto res = interpolateField({qi}, fr, n, sigma, supp, L);
  real l = L.x;
  real solution = (9278850.0* pow(M_PI,1.5)*pow(erf(0.5*l/sigma),3))/(2301620723.0*pow(sigma,3));
  real error = abs(res[0].real()-solution)/solution;
  if(error>1e-11){
    System::log<System::ERROR>("Too much error in quadrature");
    System::log<System::MESSAGE>("Error in quadrature %g, got %.15g, expected %.15g",
				 error, res[0].real(), solution);
  }
  else{
    System::log<System::MESSAGE>("[SUCCESS]");
  }

}
//Checks that F = JSF, i.e interpolating after spreading returns the original quantity.
void checkJSF(){
  System::log<System::MESSAGE>("Interpolation after spreading (quadrature) test");
  int3 n = {128,128,128};
  real l = 1;
  real3 L = {l,l,l};
  int supp = n.x;
  real3 qi = make_real3(0,0,0)*L;
  runInterpolationTest(qi, L, n, supp);
}


int main(int argc, char* argv[]){
  {
    System::log<System::MESSAGE>("Checking spreading mechanism");
    checkSpreading();
    System::log<System::MESSAGE>("Checking interpolation mechanism");
    checkJSF();
  }
  return 0;
}
