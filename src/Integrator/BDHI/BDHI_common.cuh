
/*WARNING!! ONLY INLINE AND EXTERN DEVICE FUNCTIONS IN THIS FILE. Will be compiled in each CU*/

namespace BDHI{

  struct RPYUtils{
    real rh;
    real rhrh2div3;
    real invrh3inv32;
    RPYUtils(real rh): rh(rh),
      rhrh2div3(2.0*rh*rh/3.0),
      invrh3inv32(3.0/(rh*32.0)){    }
    /*RPY tensor as a function of distance, r*/
    /*M(r) = 0.75*M0*( f(r)*I + g(r)*r(diadic)r )*/
    /*c12.x = f(r) * 0.75*M0    ->M0 is outside
      c12.y = g(r) * 0.75*M0*/
    /*This is a critical function and is insanely optimized to perform the least FLOPS possible*/
    inline __host__  __device__  real2  RPY(const real &r) const{
    
      const real invr  = real(1.0)/r;
    
      real2 c12 = {0,0};

      /*Oseen tensor*/
      //return RPYparams.rh*make_real2(real(0.75)*invr, real(0.75)*invr*invr*invr);
      /*c12.y = c2 -> c2*invr2*/
      if(r >= real(2.0)*rh){
	const real A = real(0.75)*rh*invr;
	const real invr2 = invr*invr;
	
	// c12.x = real(0.75)*(invr+(2.0f/3.0f)*rh*rh*invr2*invr)*rh;
	// c12.y = real(0.75)*(invr-2.0f*invr2*invr*rh*rh)*rh;
	
	c12.x = A*(real(1.0) + rhrh2div3*invr2);      
	c12.y = A*invr2*(real(1.0) - real(3.0)*rhrh2div3*invr2);      
      }
      else{
	c12.x = real(1.0)-(9.0f/32.0f)*r/rh;//invrh3inv32*real(3.0)*r;//
	if(r>real(0.0))
	  c12.y = (3.0f/32.0f)*invr/rh; //invrh3inv32*invr;//      
      }
      
      return c12;
    }

    /*Helper function for divergence in RDF, 
      computes {f(r+dw)-f(r), g(r+dw)-g(r)}
      See diffusionDot for more info
    */
    inline __device__ real2  RPYDivergence(real3 rij, real3 dwij) const{

      const real r    = sqrtf(dot(rij, rij));
      const real3 rpdwij = rij+dwij;
      const real rpdw = sqrtf(dot(rpdwij, rpdwij));
    
      return RPY(rpdw)-RPY(r);
    }
  };
  
  struct divMTransverser{
    divMTransverser(real3* divM, real M0, real rh): divM(divM), M0(M0), rh(rh){}
    
    inline __device__ real3 zero(){ return make_real3(real(0.0));}
    inline __device__ real3 compute(const real4 &pi, const real4 &pj){
      real3 r12 = make_real3(pi)-make_real3(pj);
      real r2 = dot(r12, r12);
      if(r2==real(0.0))
	return make_real3(real(0.0));
      /*TODO General for rh*/
      real invr = rsqrtf(r2);
      if(r2>real(1.0)){
	r2 *= real(4.0);
	return make_real3(real(0.75)*real(2.0)*(r2-real(2.0))/(r2*r2)*r12*invr);
      }
      else{
	//return make_real3(0.0);
	return make_real3(real(0.09375)*real(2.0)*r12*invr);
      }
    }
    inline __device__ void accumulate(real3 &total, const real3 &cur){total += cur;}
    
    inline __device__ void set(int id, const real3 &total){
      divM[id] = M0*total;
    }
  private:
    real3* divM;
    real M0;
    real rh;
  };

       
  
}