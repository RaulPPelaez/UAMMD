/*Raul P. Pelaez 2020. Near field section of the Doubly Periodic Poisson algorithm
*/
#ifndef DPPOISSONSLAB_NEAR_FIELD_CUH
#define DPPOISSONSLAB_NEAR_FIELD_CUH
#include "global/defines.h"
#include"Interactor/NeighbourList/CellList.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include"misc/TabulatedFunction.cuh"
namespace uammd{
  namespace DPPoissonSlab_ns{

    namespace nearField_ns{
      __host__ __device__ double GreensFunctionNearPotential(double r2, double split, double gw, double permitivity){
	const double gw2 = gw*gw;
	double G = 0;
	if(sqrt(r2)>gw*1e-2){
	  double r = sqrt(r2);
	  G = (1.0/(4.0*M_PI*permitivity*r)*(erf(r/(2*gw)) - erf(r/sqrt(4*gw2+1/(split*split)))));
	}
	else{
	  const double pi32 = pow(M_PI,1.5);
	  const double invsp2 = 1.0/(split*split);
	  const double selfterm = 1.0/(4*pi32*gw) - 1.0/(2*pi32*sqrt(4*gw2+invsp2));
	  const double r2term = 1.0/(6.0*pi32*pow(4.0*gw2 + invsp2, 1.5)) - 1.0/(48.0*pi32*gw2*gw);
	  const double r4term = 1.0/(640.0*pi32*gw2*gw2*gw) - 1.0/(20.0*pi32*pow(4*gw2+invsp2,2.5));
	  G = (selfterm + r2*r2term + r2*r2*r4term)/permitivity;
	}
	return G;
      }

      __host__ __device__ double GreensFunctionNearField(double r2, double split, double gw, double permitivity){
	double r = sqrt(r2);
	double gw2 = gw*gw;
	const double newgw = sqrt(gw2 + 1/(4.0*split*split));
	double newgw2 = newgw*newgw;
	double fmod = 0;
	if(r2==0){
	  return 0;
	}
	if(r>gw*1e-2){
	  double invrterm = exp(-0.25*r2/newgw2)/sqrt(M_PI*newgw2) - exp(-0.25*r2/gw2)/sqrt(M_PI*gw2);
	  double invr2term = erf(0.5*r/newgw) - erf(0.5*r/gw);
	  fmod += 1/(4*M_PI)*( invrterm/r - invr2term/r2);
	}
	else{
	  const double pi32 = pow(M_PI, 1.5);
	  double rterm = 1/(24*pi32)*(1.0/(gw2*gw) - 1/(newgw2*newgw));
	  double r3term = 1/(160*pi32)*(1.0/(newgw2*newgw2*newgw) - 1.0/(gw2*gw2*gw));
	  fmod += r*rterm+r2*r*r3term;
	}
	return fmod/(r*permitivity);
      }

      template<class Parameters>
      double computeCutOffDistance(Parameters par){
	real rn = par.gw;
	const real dr = par.gw*0.01;
	real ratioAtRn;
	do{
	  rn += dr;
	  real dGdr_split = GreensFunctionNearField(rn*rn, par.split, par.gw/sqrt(2.0), par.permitivity.inside);
	  real dGdr = GreensFunctionNearField(rn*rn, 1e-10, par.gw/sqrt(2.0), par.permitivity.inside);
	  ratioAtRn = abs(dGdr_split/dGdr);
	}while(ratioAtRn > par.tolerance);
	const real rcut = rn + par.numberStandardDeviations*par.gw;
	return rcut;
      }

      void throwIfInvalidConfiguration(double nearFieldCutOff, double H){
	double maximumAllowedCutOff = H;
	if(nearFieldCutOff > maximumAllowedCutOff){
	  System::log<System::EXCEPTION>("[DPPoissonSlab] Close range cut off is too large (%g, max is %g), increase splitting parameter or lower tolerance", nearFieldCutOff, maximumAllowedCutOff);
	  throw std::invalid_argument("[DPPoissonSlab] Incompatible parameters");
	}
      }

    }

    class NearField{
    public:
      struct Parameters{
	real split;
        Permitivity permitivity;
	real gw;
	real H;
	real numberStandardDeviations;
	real2 Lxy;
	real tolerance;
      };
    private:
      using NeighbourList = CellList;

      shared_ptr<System> sys;
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
      shared_ptr<NeighbourList> nl;
      Parameters par;
      real rcut;
      thrust::device_vector<real2> greensFunctionsTableData;
      std::shared_ptr<TabulatedFunction<real2>> greenTables;
      real split, gw;
    public:

      NearField(shared_ptr<System> sys, shared_ptr<ParticleData> pd, shared_ptr<ParticleGroup> pg,
		Parameters par):sys(sys), pd(pd), pg(pg), par(par), split(par.split), gw(par.gw){	
	this->rcut = nearField_ns::computeCutOffDistance(par);
	nearField_ns::throwIfInvalidConfiguration(rcut, par.H);
	sys->log<System::MESSAGE>("[DPPoissonSlab] Near field cut off: %g", rcut);
	initializeTabulatedGreensFunctions();
      }

      void initializeTabulatedGreensFunctions();

      void compute(cudaStream_t st);

    };

    void NearField::initializeTabulatedGreensFunctions(){
      //TODO: I need a better heuristic to select the table size
      int Ntable = std::max(1<<16, std::min(1<<20, 2*int(rcut/(par.gw*par.tolerance*1e2))));
      sys->log<System::MESSAGE>("[Poisson] Elements in near field table: %d", Ntable);
      greensFunctionsTableData.resize(Ntable);
      real2* ptr = thrust::raw_pointer_cast(greensFunctionsTableData.data());
      greenTables = std::make_shared<TabulatedFunction<real2>>(ptr, Ntable, 0, rcut*rcut,
	[=](real r2){
	  real Gpotential = nearField_ns::GreensFunctionNearPotential(r2, par.split, par.gw, par.permitivity.inside);
	  real Gfield = nearField_ns::GreensFunctionNearField(r2, par.split, par.gw, par.permitivity.inside);
	  return make_real2(Gpotential, Gfield);
	});
    }

    struct NearFieldTransverser{
      using returnInfo = real4;

      NearFieldTransverser(real* energy_ptr, real4* force_ptr, real* charge,
			   TabulatedFunction<real2> greenTables,
			   Box box, real H, real rcut, Permitivity perm, real split, real gw):
	energy_ptr(energy_ptr), force_ptr(force_ptr), charge(charge),split(split), gw(gw),
        box(box), H(H), rcut(rcut), perm(perm), GreensFunctionFieldAndPotential(greenTables){}

      __device__ returnInfo zero() const{ return returnInfo();}

      struct Info{
	real charge;
	int i;
      };
      __device__ Info getInfo(int pi) const{ return {charge[pi], pi};}

      __device__ returnInfo compute(real4 pi, real4 pj, Info infoi, Info infoj) const{
	auto FandE = infoj.charge*computeFieldPotential(make_real3(pi), make_real3(pj));
	 const bool hasImageCharge = abs(abs(pj.z) - H*real(0.5)) < rcut;
	 if(hasImageCharge){
	   //Nearest image
	   real3 pjim = make_real3(pj);
	   pjim.z = (pj.z>0?H:-H) - pj.z;
	   real ep = pj.z<0?perm.bottom:perm.top;
	   real epratio = isinf(ep)?real(1.0):(ep - perm.inside) / (ep + perm.inside);
           real chargeImage = -infoj.charge * epratio;
	   FandE += chargeImage*computeFieldPotential(make_real3(pi), pjim);
	   //Image in the opposite side
	   if(rcut >= H*real(0.5) ){
	     pjim = make_real3(pj);
	     pjim.z = (pj.z>0?-H:H) - pj.z;
	     ep = pj.z<0?perm.top:perm.bottom;
	     epratio = isinf(ep)?real(1.0):(ep - perm.inside) / (ep + perm.inside);
	     chargeImage = -infoj.charge * epratio;
	     FandE += chargeImage*computeFieldPotential(make_real3(pi), pjim);
	   }
         }
	return infoi.charge*FandE;
      }

      __device__ void accumulate(returnInfo &total, returnInfo current) const {total += current;}

      __device__ void set(uint pi, returnInfo total) const {
	force_ptr[pi] += make_real4(make_real3(total), 0);
	//energy_ptr[pi] += total.w;
      }

    private:
      real* energy_ptr;
      real4* force_ptr;
      real* charge;
      Box box;
      real H, rcut;
      Permitivity perm;
      real split;
      real gw;
      TabulatedFunction<real2> GreensFunctionFieldAndPotential;

      __device__ returnInfo computeFieldPotential(real3 pi, real3 pj) const{
	real3 rij = box.apply_pbc(pi-pj);
        real r2 = dot(rij, rij);
	if(r2 >= rcut*rcut) return real4();
        real2 greensFunctions = GreensFunctionFieldAndPotential(r2);
	// real2 greensFunctions = make_real2(nearField_ns::GreensFunctionNearPotential(r2, split, gw, perm.inside),
	// 				   nearField_ns::GreensFunctionNearField(r2, split, gw, perm.inside));
	real potential = greensFunctions.x;
	real3 field = greensFunctions.y*rij;
	return make_real4(field, potential);
      }

    };

    void NearField::compute(cudaStream_t st){
      if(par.split){
	sys->log<System::DEBUG2>("[DPPoissonSlab] Near field energy computation");
	Box box(make_real3(par.Lxy, par.H));
	box.setPeriodicity(1, 1, 0);
	if(!nl){
	  nl = std::make_shared<NeighbourList>(pd, pg, sys);
	}
	nl->update(box, rcut, st);
	auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
	auto charge = pd->getCharge(access::location::gpu, access::mode::read);
	auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
        auto tr = NearFieldTransverser(energy.begin(), force.begin(), charge.begin(),
				       *greenTables, box, par.H, rcut, par.permitivity, split, gw);
	nl->transverseList(tr, st);
	CudaCheckError();
      }
    }

  }
}
#endif
