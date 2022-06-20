/* Raul P. Pelaez 2022. Compressible ICM test code.
   Can compute dynamic structure factors for the fluid as well as the VACF (Velocity autocorrelation function).
 */
#include "uammd.cuh"
#include "computeStructureFactor.cuh"
#include "Integrator/Hydro/ICM_Compressible.cuh"
#include "utils/container.h"
#include "utils/InputFile.h"
#include <cstdint>

using namespace uammd;

using ICM = Hydro::ICM_Compressible;


struct Parameters{
  real dt = 0.1;
  real3 boxSize = make_real3(32,32,32)*100;
  int3 cellDim = {30,30,30};
  real bulkViscosity = 127.05;
  real speedOfSound = 14.67;
  real shearViscosity = 53.71;
  real temperature = 1;
  real initialDensity = 0.632;

  real relaxTime = 500;
  int numberFreq = 200;
  real maxFreq = -1;
  real simulationTime = -1;

  int3 waveNumber = {2,2,2};

  bool measureVACF = false;
};

auto createICMIntegrator(std::shared_ptr<ParticleData> pd, Parameters ipar){
  ICM::Parameters par;
  par.dt = ipar.dt;
  par.boxSize = ipar.boxSize;
  //par.hydrodynamicRadius = 0.5;
  par.cellDim = ipar.cellDim;
  par.bulkViscosity = ipar.bulkViscosity;
  par.speedOfSound = ipar.speedOfSound;
  par.shearViscosity = ipar.shearViscosity;
  par.temperature = ipar.temperature;
  par.initialDensity = [=](real3 r){return ipar.initialDensity;};
  return std::make_shared<ICM>(pd, par);
}

Parameters readParameters(std::string file){
  InputFile in(file);
  Parameters par;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("boxSize", InputFile::Required)>>par.boxSize.x>>par.boxSize.y>>par.boxSize.z;
  in.getOption("cellDim", InputFile::Required)>>par.cellDim.x>>par.cellDim.y>>par.cellDim.z;
  in.getOption("bulkViscosity", InputFile::Required)>>par.bulkViscosity;
  in.getOption("shearViscosity", InputFile::Required)>>par.shearViscosity;
  in.getOption("speedOfSound", InputFile::Required)>>par.speedOfSound;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  in.getOption("initialDensity", InputFile::Required)>>par.initialDensity;

  in.getOption("relaxTime", InputFile::Optional)>>par.relaxTime;
  in.getOption("numberFreq", InputFile::Optional)>>par.numberFreq;
  in.getOption("maxFreq", InputFile::Optional)>>par.maxFreq;
  in.getOption("simulationTime", InputFile::Optional)>>par.simulationTime;
  in.getOption("waveNumber", InputFile::Optional)>>par.waveNumber;
  if(bool(in.getOption("measureVACF"))){
    par.measureVACF = true;
  }
  return par;
}

real indexToWaveVectorModulus(int i, int3 nk, real3 L){
  int kx = i%(nk.x/2+1);
  int ky = (i/(nk.x/2+1))%nk.y;
  int kz = i/((nk.x/2+1)*nk.y);
  real3 kvec = 2*M_PI*make_real3(kx,ky,kz)/L;
  return sqrt(dot(kvec, kvec));
}

using complex = FourierTransform3D::complex;
template<class Container1, class Container2>
auto filterSamples(Container1 &v, Container2 samples){
  int nsamples = samples.size();
  uninitialized_cached_vector<complex> usamples(nsamples);
  auto perm = thrust::make_permutation_iterator(v.begin(), samples.begin());
  thrust::copy(perm, perm + nsamples, usamples.begin());
  return usamples;
}

class VelocityAutocorrelation {
  int nbin = 100;
  std::vector<int> histogram;
  real maxv = 0.005;
  int size;
  int nsamples = 0;
public:
  VelocityAutocorrelation(){
    histogram = std::vector<int>(nbin,0);
  }

  template<class Container>
  void addSample(Container &vel){
    nsamples++;
    size = vel.size();
    std::vector<real> h_vx(size);
    //real maxv = *thrust::max_element(vel.x(), vel.x() + vel.size());
    thrust::copy(vel.x(), vel.x() + size, h_vx.begin());
    for(auto v: h_vx){
      int bin = nbin*(v+maxv)/(maxv + maxv);
      if(bin < nbin and bin >= 0){
	histogram[bin]++;
      }
    }
  }

  void write(){
    std::ofstream out("vacf.dat");
    double dx = (2*maxv)/nbin;
    fori(0, nbin){
      double x = (i+0.5)/nbin*2*maxv - maxv;
      out<<x<<" "<<histogram[i]/(size*dx*nsamples)<<"\n";
    }
  }
};

template<class Container1, class Container2>
void writeComplexSignals(Container1 &Sqw, Parameters par, int ntimes, Container2 &sampleWaveNumbers, std::string name){
  const int nsamples = sampleWaveNumbers.size();
  std::vector<complex> Sqw_h(Sqw.size());
  thrust::copy(Sqw.begin(), Sqw.end(), Sqw_h.begin());
  //  real T = ntimes*par.dt;
  //  real V = par.boxSize.x*par.boxSize.y*par.boxSize.z;
  fori(0,nsamples){
    int isample = sampleWaveNumbers[i];
    std::ofstream out(name+"qw"+std::to_string(isample)+".dat");
    real q = indexToWaveVectorModulus(isample, par.cellDim, par.boxSize);
    forj(0, par.numberFreq){
      out<<2*M_PI*j/(par.simulationTime)<<" "<<Sqw_h[ntimes*i + j]<<"\n";
    }
  }
}

int main(int argc, char *argv[]){
  auto sys = std::make_shared<System>(argc, argv);
  auto pd = std::make_shared<ParticleData>(0, sys);
  auto par = readParameters(argv[1]);
  auto icm = createICMIntegrator(pd, par);
  if(par.maxFreq<0) par.maxFreq = 4*par.speedOfSound*2*M_PI*2/par.boxSize.x;
  if(par.simulationTime<0) par.simulationTime= 2*M_PI*par.numberFreq/par.maxFreq;
  {
    int relaxSteps =par.relaxTime/par.dt+1;
    fori(0, relaxSteps) icm->forwardTime();
  }
  int ntimes = par.simulationTime/par.dt;
  System::log<System::MESSAGE>("Number of steps: %d", ntimes);
  int nsamples = 1;
  thrust::device_vector<uint64_t> sampleWaveNumbers(nsamples);
  //for(auto s: sampleWaveNumbers) s = pd->getSystem()->rng().next32()%nk;
  // int count = 0;
  // for(auto s: sampleWaveNumbers) s = count++;
  {
    auto kn = par.waveNumber;
    sampleWaveNumbers[0] = kn.x + (kn.y + kn.z*par.cellDim.y)*(par.cellDim.x/2+1);
  }
  FourierTransform3D fft(par.cellDim);
  // DynamicStructureFactor Sqw_vxvy(nsamples, ntimes), Sqw_vxvx(nsamples, ntimes);
  // DynamicStructureFactor Sqw_rhovx(nsamples, ntimes), Sqw_rhorho(nsamples, ntimes);
  FourierTransformComplex1D vx_kw(nsamples, ntimes), vy_kw(nsamples, ntimes);
  FourierTransformComplex1D rho_kw(nsamples, ntimes);
  VelocityAutocorrelation vacf;
  int sampleSteps = 1;
  fori(0, ntimes){
    icm->forwardTime();
    if(i%sampleSteps == 0){
      auto dens = icm->getCurrentDensity();
      // std::vector<real> h_d(dens.size());
      // thrust::copy(dens.begin(), dens.end(), h_d.begin());
      // real averageDens = std::accumulate(dens.begin(), dens.end(), 0.0)/dens.size();
      //System::log<System::MESSAGE>("Average density: %g", averageDens);
      auto vel = icm->getCurrentVelocity();
      if(par.measureVACF)
	vacf.addSample(vel);
      auto vxq = fft.transform(vel.x());
      auto vyq = fft.transform(vel.y());
      auto rhoq = fft.transform(thrust::raw_pointer_cast(dens.data()));
      auto vxsamples = filterSamples(vxq, sampleWaveNumbers);
      auto vysamples = filterSamples(vyq, sampleWaveNumbers);
      auto rhosamples = filterSamples(rhoq, sampleWaveNumbers);
      vx_kw.addSamplesFourier(thrust::raw_pointer_cast(vxsamples.data()), i );
      vy_kw.addSamplesFourier(thrust::raw_pointer_cast(vysamples.data()), i );
      rho_kw.addSamplesFourier(thrust::raw_pointer_cast(rhosamples.data()), i );
      // Sqw_vxvy.addSamplesFourier(thrust::raw_pointer_cast(vxsamples.data()), thrust::raw_pointer_cast(vysamples.data()), i);
      // Sqw_vxvx.addSamplesFourier(thrust::raw_pointer_cast(vxsamples.data()), thrust::raw_pointer_cast(vxsamples.data()), i);
      // Sqw_rhovx.addSamplesFourier(thrust::raw_pointer_cast(rhosamples.data()), thrust::raw_pointer_cast(vxsamples.data()), i);
      // Sqw_rhorho.addSamplesFourier(thrust::raw_pointer_cast(rhosamples.data()), thrust::raw_pointer_cast(rhosamples.data()), i);
    }
  }
  System::log<System::MESSAGE>("Processing and writing results");
  if(par.measureVACF)
    vacf.write();
  auto vx = vx_kw.compute();
  writeComplexSignals(vx, par, ntimes, sampleWaveNumbers, "vx");
  auto vy = vy_kw.compute();
  writeComplexSignals(vy, par, ntimes, sampleWaveNumbers, "vy");
  auto rho = rho_kw.compute();
  writeComplexSignals(rho, par, ntimes, sampleWaveNumbers, "rho");
  // auto vxvy = Sqw_vxvy.compute();
  // writeComplexSignals(vxvy, par, ntimes, sampleWaveNumbers, "Svxvy");
  // auto vxvx = Sqw_vxvx.compute();
  // writeComplexSignals(vxvx, par, ntimes, sampleWaveNumbers, "Svxvx");
  // auto rhovx = Sqw_rhovx.compute();
  // writeComplexSignals(rhovx, par, ntimes, sampleWaveNumbers, "Srhovx");
  // auto rhorho = Sqw_rhorho.compute();
  // writeComplexSignals(rhorho, par, ntimes, sampleWaveNumbers, "Srhorho");
  return 0;
}
