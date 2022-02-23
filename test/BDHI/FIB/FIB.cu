/*Raul P. Pelaez 2018. BDHI::FIB tests

All output is adimensional.



 */
#include"uammd.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Interactor/Interactor.cuh"
#include"Integrator/BDHI/FIB.cuh"

#include<iostream>

#include<vector>
#include<fstream>
#include<iomanip>
using namespace uammd;

real temperature, viscosity, rh, tolerance;

//RPY kernel M(\vec{r}) = f(r)·I + g(r)·\vec{r}\otimes\vec{r}/r^2
//M0 = f(0)
long double f(long double r, long double a = rh){
  long double M0 = 1/(6*M_PIl*viscosity*a);
  if(r>2*a)
    return M0*(3*a/(4*r) + (a*a*a*0.5/(r*r*r)));
  else
    return M0*(1-9*r/(32*a));
}
long double g(long double r, long double a = rh){
  long double M0 = 1/(6*M_PIl*viscosity*a);
  if(r>2*a)
    return M0*(3*a/(4*r) - 1.5*a*a*a/(r*r*r));
  else
    return M0*(3*r/(32.0*a));
}


//Pulls two particles agains each other, or just the first one if there is only one particle
class miniInteractor: public Interactor{
public:
  using Interactor::Interactor;
  real3 F;
  void sum(Computables comp, cudaStream_t st) override{
    auto force = pd->getForce(access::location::cpu, access::mode::write);
    force.raw()[0] = make_real4(F,0);
    if(pg->getNumberParticles()>1)
      force.raw()[1] = make_real4(real(-1.0)*F,0);
  }
};

using std::make_shared;
using std::endl;
//Self mobility deterministic test. Pull a particle with a force, measure its velocity.
void computeSelfMobilityMatrix(real3 L, double F, long double *M, double &true_rh, double &true_M0){
  int N = 1;
  auto sys = make_shared<System>();
  sys->rng().setSeed(0xabefa129f9173^time(NULL));
  for(int i = 0; i<1000; i++) sys->rng().next();
  auto pd = make_shared<ParticleData>(N, sys);
  
  Box box(L);
  BDHI::FIB::Parameters par;
  par.temperature = 0.0;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 0.01;
  par.box = box;
  auto bdhi = make_shared<BDHI::FIB>(pd, par);
  true_M0 = bdhi->getSelfMobility();
  true_rh = bdhi->getHydrodynamicRadius();
  auto inter= make_shared<miniInteractor>(pd, "puller");
  bdhi->addInteractor(inter);
  for(int i = 0; i<9;i++){M[i] = 0;}
  int Ntest = 10;
  for(int i = 0; i<Ntest;i++){
    for(int alpha = 0; alpha<3;alpha++){
      double3 posprev;
      {
	auto pos = pd->getPos(access::location::cpu, access::mode::write);
	pos.raw()[0] = make_real4(make_real3(sys->rng().uniform3(-0.5, 0.5))*box.boxSize,0);
	posprev = make_double3(make_real3(pos.raw()[0]));
      }
      inter->F = F*make_real3(alpha==0, alpha==1, alpha==2);
      bdhi->forwardTime();
      double3 vel;
      {
	auto pos = pd->getPos(access::location::cpu, access::mode::read);
	vel = (make_double3(make_real3(pos.raw()[0]))-posprev)/par.dt;
      }
      M[alpha+3*0] += vel.x/(F*double(Ntest));
      M[alpha+3*1] += vel.y/(F*double(Ntest));
      M[alpha+3*2] += vel.z/(F*double(Ntest));
    }
  }
  sys->finish();
}

bool selfMobilityCubicBox_test(){
  int NL = 30;
  std::vector<real2> velocities(NL);
  real L_min = 8*rh;
  real L_max = 128*rh;
  double F = 1;
  long double M[9];
  std::ofstream Mout("selfMobilityCubicBox.test");
  fori(0, NL){
    double L = L_min + i*((L_max-L_min)/(real)(NL-1));
    double true_rh;
    double M0;
    computeSelfMobilityMatrix(make_real3(L), F, M, true_rh, M0);
    CudaCheckError();
    Mout<<std::setprecision(15)<<L/true_rh<<" ";
    //Substract 1 to the diagonal terms, which should be one so a matrix of zeroes should be printed
    //abs to be able to plot log
    for(int j=0; j<9; j++) Mout<<std::setprecision(15)<<abs((1*(j%3==j/3)-M[j]/M0))<<" ";
    Mout<<endl;
  }
  return true;
}

void computePairMobilityMatrix(real3 L, double F, double3 dist, long double *M, double &true_rh){
  int N = 2;
  auto sys = make_shared<System>();
  sys->rng().setSeed(0xabefa129f9173^time(NULL));
  for(int i = 0; i<10000; i++) sys->rng().next();
  auto pd = make_shared<ParticleData>(N, sys);
  

  Box box(L);
  BDHI::FIB::Parameters par;
  par.temperature = 0.0;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 0.01;
  par.box = box;


  auto bdhi = make_shared<BDHI::FIB>(pd, par);

  double M0 = bdhi->getSelfMobility();
  true_rh = bdhi->getHydrodynamicRadius();
  auto inter= make_shared<miniInteractor>(pd, "puller");
  bdhi->addInteractor(inter);


  for(int i = 0; i<9;i++){M[i] = 0;}
  int Ntest = 30;
  for(int i = 0; i<Ntest;i++){
    for(int alpha = 0; alpha<3;alpha++){
      double3 posprev;
      {
	auto pos = pd->getPos(access::location::cpu, access::mode::write);
	double3 ori = sys->rng().uniform3(-0.5, 0.5)*make_double3(box.boxSize);
	pos.raw()[0] = make_real4(ori,0);
	pos.raw()[1] = make_real4(ori+dist,0);
	posprev = make_double3(make_real3(pos.raw()[1]));
      }

      inter->F = F*make_real3(alpha==0, alpha==1, alpha==2);
      bdhi->forwardTime();
      double3 vel;
      {
	auto pos = pd->getPos(access::location::cpu, access::mode::read);
	vel = (make_double3(make_real3(pos.raw()[1]))-posprev)/par.dt;
      }
      M0=1;
      M[alpha+3*0] += vel.x/(F*M0*double(Ntest));
      M[alpha+3*1] += vel.y/(F*M0*double(Ntest));
      M[alpha+3*2] += vel.z/(F*M0*double(Ntest));
    }
  }
  sys->finish();
}

bool pairMobilityCubicBox_test(double dist){

  int NL = 20;

  real L_min = 2.1*dist;
  real L_max = 256*rh;

  Xorshift128plus rng;
  rng.setSeed(0x12ffdbae328f01^time(NULL));
  for(int i = 0; i<10000; i++) rng.next();
  real3 dir = make_real3(0);

  double3 rij;
  do{
    dir = make_real3(0);
    while(dir.x == 0 or dir.y == 0 or dir.z == 0) dir = make_real3(rng.gaussian3(0,1));
    rij = dist*make_double3(dir)/sqrt(dot(dir,dir));
    std::cerr<<rij.x<<endl;
  }while(rij.x<1 or rij.y<1 or rij.z<1);
  std::ofstream out("pairMobilityCubicBox.dist"+std::to_string(dist/rh)+".test");
  double F = 1;
  long double M[9];

  double M_theo_Linf[9];
  out<<"#rij "<<rij.x<<" "<<rij.y<<" "<<rij.z<<endl;
  fori(0, NL){
    real L = L_min + i*((L_max-L_min)/(real)(NL-1));

    double true_rh;
    computePairMobilityMatrix(make_real3(L), F, rij, M, true_rh);

    for(int i=0; i<9; i++){ M_theo_Linf[i] = 0;}
    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	long double r = sqrt(dot(rij, rij));
	double *r01 = &rij.x;
	long double diadic = 0;
	if(r>0) diadic = r01[i]*r01[j]/(r*r);
	M_theo_Linf[3*i+j] = g(r,true_rh)*diadic;
	long double fr = f(r,true_rh)-f(0,true_rh);
	if(i==j) M_theo_Linf[3*i+j] += fr;
      }
    }

    out<<std::setprecision(15)<<L/true_rh<<" ";

    for(int j = 0; j<9; j++)    out<<abs(1.0l-M[j]/(M_theo_Linf[j]))<<" ";
    out<<endl;

    CudaCheckError();
  }
  return true;
}

bool pairMobility_q2D_test(double dist){

  int NL = 400;

  real L_min = 2.1*dist;
  real L_max = 200*rh;

  Xorshift128plus rng;
  rng.setSeed(0x12ffdbae328f01^time(NULL));
  for(int i = 0; i<10000; i++) rng.next();
  real3 dir = make_real3(0);

  while(dir.x == 0 or dir.y == 0 or dir.z == 0) dir = make_real3(rng.gaussian3(0,1));
  double3 rij = dist*make_double3(dir)/sqrt(dot(dir,dir));

  std::ofstream out("pairMobility_q2D.dist"+std::to_string(dist/rh)+".test");
  double F = 1;
  long double M[9];

  out<<"#rij "<<rij.x<<" "<<rij.y<<" "<<rij.z<<endl;
  real Lx = 32*rh;
  fori(0, NL){
    real Lz = L_min + i*((L_max-L_min)/(real)(NL-1));

    double true_rh;
    computePairMobilityMatrix(make_real3(Lx,Lx,Lz), F, rij, M, true_rh);
    out<<std::setprecision(15)<<Lz/true_rh<<" ";

    double M0 = 1.0/(6*M_PI*viscosity*true_rh);
    for(int j = 0; j<9; j++)  out<<M[j]/M0<<" ";
    out<<endl;

    CudaCheckError();
  }
  return true;
}

bool selfMobility_q2D_test(){

  int NL = 40;

  real L_min = 8*rh;
  real L_max = 200*rh;

  double F = 1;
  long double M[9];
  std::ofstream Mout("selfMobility_q2D.test");
  std::ofstream Mtheoout("selfMobility_q2D.theo.test");
  fori(0, NL){
    real Lz = L_min + i*((L_max-L_min)/(real)(NL-1));
    real L = 32*rh;
    double true_rh, true_M0;

    computeSelfMobilityMatrix(make_real3(L,L, Lz), F, M, true_rh, true_M0);

    CudaCheckError();

    long double M0 = 1.0L/(6.0L*M_PIl*viscosity*true_rh);
    //From eq 21 and 23 in Vögele, M., & Hummer, G. (2016). Divergent Diffusion Coefficients in Simulations of Fluids and Lipid Membranes. The Journal of Physical Chemistry B, 120(33), 8722–8732. doi:10.1021/acs.jpcb.6b05102

    double Mplane_near = M0 + M0*true_rh/L*(M_PI*0.5*Lz/L - 4.3878);
    double Mplane_far = M0 + M0*true_rh/Lz*(1.5*log(L/Lz) - 2.8897);

    double Mperp_near = M0 + M0*true_rh/Lz*(3*log(L/Lz) - 2.77939);
    double Mperp_far = M0 - M0*2.9252*true_rh/L;

    Mout<<std::setprecision(15)<<Lz/true_rh<<" ";
    Mtheoout<<std::setprecision(15)<<Lz/true_rh<<" ";

    for(int j=0; j<9; j++) Mout<<std::setprecision(15)<<M[j]/M0<<" ";

    Mtheoout<<std::setprecision(15)<<Mplane_near/M0<<" "<<Mplane_far/M0<<" "<<Mperp_near/M0<<" "<<Mperp_far/M0<<endl;
    Mout<<endl;
  }

  return true;
}


bool idealParticlesDiffusion(int N, real3 L, double &true_rh, std::string suffix = "test"){
  auto sys = make_shared<System>();
  sys->rng().setSeed(0x33dbff9f235ab^time(NULL));
  for(int i=0; i<10000; i++) sys->rng().next();

  auto pd = make_shared<ParticleData>(N, sys);
  

  Box box(L);
  BDHI::FIB::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 0.001;
  par.box = box;


  auto bdhi = make_shared<BDHI::FIB>(pd, par);
  true_rh = bdhi->getHydrodynamicRadius();
  std::ofstream out("pos.noise.boxSize"+std::to_string(L.z/true_rh)+".rh"+std::to_string(true_rh)+".dt"+std::to_string(par.dt)+"."+suffix);
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    fori(0, pd->getNumParticles()){
      pos.raw()[i] = make_real4(make_real3(sys->rng().uniform3(-0.5, 0.5))*L, 0);
    }
  }

  fori(0,2000){
    bdhi->forwardTime();
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
    real4 *p = pos.raw();
    out<<"#"<<endl;
    forj(0,pd->getNumParticles()){
      out<<std::setprecision(15)<<make_real3(p[j])<<"\n";
    }
  }

  sys->finish();
  return true;
}

void selfDiffusionCubicBox_test(){
  int NL = 5;
  real L_min = 8*rh;
  real L_max = 128*rh;
  int N=4096;
  forj(0, NL){
    real L = L_min + j*((L_max-L_min)/(real)(NL-1));
    double true_rh;
    idealParticlesDiffusion(N, make_real3(L),true_rh);
    CudaCheckError();
  }


}

void selfDiffusion_q2D_test(){
  int NL = 20;
  real L_min = 8*rh;
  real L_max = 128*rh;

  real Lx = 32*rh;
  int N=4096;
  std::ofstream out("selfDiffusion_q2D.theo");
  forj(0, NL){
    double Lz = L_min + j*((L_max-L_min)/(real)(NL-1));
    double true_rh;
    idealParticlesDiffusion(N, make_real3(Lx, Lx, Lz), true_rh, "q2D.Lx"+std::to_string(Lx/rh)+".test");
    double L = Lx;
    double lz = Lz;
    long double M0 = 1.0L/(6.0L*M_PIl*viscosity*true_rh);
    //From eq 21 and 23 in Vögele, M., & Hummer, G. (2016). Divergent Diffusion Coefficients in Simulations of Fluids and Lipid Membranes. The Journal of Physical Chemistry B, 120(33), 8722–8732. doi:10.1021/acs.jpcb.6b05102

    double Mplane_near = M0 + M0*true_rh/L*(M_PI*0.5*lz/L - 4.3878);
    double Mplane_far = M0 + M0*true_rh/lz*(1.5*log(L/lz) - 2.8897);

    double Mperp_near = M0 + M0*true_rh/lz*(3*log(L/lz) - 2.77939);
    double Mperp_far = M0 - 2.9252/(6*M_PI*viscosity*L);

    out<<std::setprecision(15)<<Lz/rh<<" "<<Mplane_near/M0<<" "<<Mplane_far/M0<<" ";
    out<<std::setprecision(15)<<Mperp_near/M0<<" "<<Mperp_far/M0<<endl;

    CudaCheckError();
  }


}


//Returns Var(noise)
double3 singleParticleNoise(real T, real3 L, double &true_rh){
  int N = 1;
  auto sys = make_shared<System>();
  sys->rng().setSeed(1234791);
  auto pd = make_shared<ParticleData>(N, sys);
  

  Box box(L);
  BDHI::FIB::Parameters par;
  par.temperature = T;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 1.0;
  par.box = box;


  auto bdhi = make_shared<BDHI::FIB>(pd, par);
  true_rh = bdhi->getHydrodynamicRadius();
  double3 prevp;
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos.raw()[0] = make_real4(make_real3(sys->rng().uniform3(-0.5, 0.5))*box.boxSize, 0);
    prevp = make_double3(pos.raw()[0]);
  }
  double3 variance = make_double3(0);
  double3 mean = make_double3(0);
  int nsteps = 10000;
  fori(0,nsteps){
    bdhi->forwardTime();
    auto pos = pd->getPos(access::location::cpu, access::mode::read);
    real4 *p = pos.raw();
    double3 noise = make_double3(p[0]) - prevp;
    double3 delta = noise - mean;
    mean += delta/double(i+1);
    double3 delta2 = noise - mean;
    variance += delta*delta2;
    prevp = make_double3(p[0]);
  }
  variance /= real(nsteps);

  sys->finish();

  return variance/(2*T*bdhi->getSelfMobility());
}


void noiseVariance_test(){
  int NL = 10;
  real L_min = 4.0*rh;
  real L_max = 128.0*rh;

  real T = temperature;
  std::ofstream out("noiseVariance.test");

  forj(0, NL){
    real L = L_min + j*((L_max-L_min)/(real)(NL-1));
    double true_rh;
    double3 noiseCorr = singleParticleNoise(T, make_real3(L), true_rh);

    //This should be close to 1 in the three directions
    out<<std::setprecision(15)<<L/true_rh<<" "<<noiseCorr.x<<" "<<noiseCorr.y<<" "<<noiseCorr.z<<endl;
    CudaCheckError();
  }

}


void radialDistributionFunction_test(){
  auto sys = make_shared<System>();
  sys->rng().setSeed(0x33dbff9f235ab^time(NULL));
  for(int i=0; i<10000; i++) sys->rng().next();
  int N = 8192*2;
  double L = 32;

  auto pd = make_shared<ParticleData>(N, sys);
  

  Box box(L);
  BDHI::FIB::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = rh;
  par.dt = 0.01;
  par.box = box;


  auto bdhi = make_shared<BDHI::FIB>(pd, par);


  std::ofstream out("rdf.pos");
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    fori(0, pd->getNumParticles()){
      pos.raw()[i] = make_real4(make_real3(sys->rng().uniform3(-0.5, 0.5))*L, 0);
    }
  }

  int nsteps = 1000000;
  int printSteps = 1000;
  fori(0,nsteps){
    bdhi->forwardTime();
    if(i%printSteps == 0){
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      real4 *p = pos.raw();
      out<<"#"<<endl;
      forj(0,pd->getNumParticles()){
	out<<std::setprecision(15)<<make_real3(p[j])<<"\n";
      }
    }
  }

  sys->finish();


}
using namespace std;
int main( int argc, char *argv[]){

  temperature = std::stod(argv[2]);
  viscosity = std::stod(argv[3]);
  rh = std::stod(argv[4]);
  tolerance = std::stod(argv[5]);
  if(strcmp(argv[1], "selfMobilityCubicBox")==0) selfMobilityCubicBox_test();
  if(strcmp(argv[1], "pairMobilityCubicBox")==0){
    pairMobilityCubicBox_test(16*rh);
    pairMobilityCubicBox_test(4*rh);
    pairMobilityCubicBox_test(6*rh);

  }
  if(strcmp(argv[1], "pairMobility_q2D")==0){
    pairMobility_q2D_test(4*rh);
    pairMobility_q2D_test(6*rh);
    pairMobility_q2D_test(8*rh);
  }

  if(strcmp(argv[1], "selfMobility_q2D")==0) selfMobility_q2D_test();
  if(strcmp(argv[1], "selfDiffusionCubicBox")==0) selfDiffusionCubicBox_test();
  if(strcmp(argv[1], "selfDiffusion_q2D")==0) selfDiffusion_q2D_test();
  if(strcmp(argv[1], "noiseVariance")==0) noiseVariance_test();
  if(strcmp(argv[1], "radialDistributionFunction")==0) radialDistributionFunction_test();
  return 0;
}
