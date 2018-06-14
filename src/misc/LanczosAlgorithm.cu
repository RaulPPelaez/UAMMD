/*Raul P. Pelaez 2017. Lanczos algorithm

References:
  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations.
  -http://dx.doi.org/10.1063/1.4742347

*/
#include"LanczosAlgorithm.cuh" 



#ifdef USE_MKL
#include<mkl.h>
#else
#include<lapacke.h>
#include<cblas.h>
#endif

#include"lapack_and_blas_defines.h"
#include"utils/debugTools.cuh"
#include"utils/cublasDebug.h"

#include<thrust/device_vector.h>
#include<fstream>

namespace uammd{
  void LanczosAlgorithm::init(){
    //Init cuBLAS for Lanczos process
    CublasSafeCall(cublasCreate_v2(&cublas_handle));
    sys->log<System::DEBUG1>("[LanczosAlgorithm] Success!");  
  }

  LanczosAlgorithm::LanczosAlgorithm(shared_ptr<System> sys, real tolerance):
    sys(sys),
    N(0),
    max_iter(3), check_convergence_steps(3)
  {    
    sys->log<System::DEBUG1>("[LanczosAlgorithm] Initializing");
    //Allocate necessary startig space
    this->increment_max_iter(0);
    this->init();
  }

  void LanczosAlgorithm::numElementsChanged(int newN){
    sys->log<System::DEBUG3>("[LanczosAlgorithm] Number of elements changed.");  
    this-> N = newN;
    w.resize(N+1, real3());
    V.resize(3*N*max_iter, 0);
    oldBz.resize(N+1, real3());  
  }
  //Increase maximum dimension of Krylov subspace, reserve necessary memory
  void LanczosAlgorithm::increment_max_iter(int inc){
    sys->log<System::DEBUG3>("[LanczosAlgorithm] Increasing subspace dimension.");  
    
    V.resize(3*N*(max_iter+inc),0);
    P.resize((max_iter+inc)*(max_iter+inc),0);
    
    hdiag.resize((max_iter+inc)+1,0);
    hsup.resize((max_iter+inc)+1,0);
    htemp.resize(2*(max_iter+inc),0);
    htempGPU.resize(2*(max_iter+inc),0);
    
    
    this->max_iter += inc;
  }

  //After a certain number of iterations (iter), computes the current result guess sqrt(M)·v, stores in BdW
  void LanczosAlgorithm::compResult(real z2, int N, int iter, real * BdW, cudaStream_t st){
    sys->log<System::DEBUG3>("[LanczosAlgorithm] Computing result");
    iter++;
    real alpha = 1.0;
    real beta = 0.0;
    /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/
    /**** H^1/2·e1 = Pt· first_column_of(sqrt(Hdiag)·P) ******/
  
    /**************LAPACKE********************/

    /*The tridiagonal matrix is stored only with its diagonal and subdiagonal*/
    /*Store both in a temporal array*/
    fori(0,iter){
      htemp[i] = hdiag[i];
      htemp[i+iter]= hsup[i];
    }
    /*P = eigenvectors must be filled with zeros, I do not know why*/
    real* h_P = thrust::raw_pointer_cast(P.data());
    memset(h_P, 0, iter*iter*sizeof(real));
  
    /*Compute eigenvalues and eigenvectors of a triangular symmetric matrix*/
    auto info = LAPACKE_steqr(LAPACK_COL_MAJOR, 'I',
			      iter, &htemp[0], &htemp[0]+iter,
			      h_P, iter);
    if(info!=0){
      sys->log<System::CRITICAL>("[LanczosAlgorithm] Could not diagonalize tridiagonal krylov matrix, steqr failed with code %d", info);      
    }

    /***Hdiag_temp = Hdiag·P·e1****/
    forj(0,iter){
      htemp[j] = sqrt(htemp[j])*P[iter*j];
    }


    /***** Htemp = H^1/2·e1 = Pt· hdiag_temp ****/
    /*Compute with blas*/
    cblas_gemv(CblasColMajor, CblasNoTrans,
	       iter, iter,
	       alpha,
	       h_P, iter,
	       &htemp[0], 1,
	       beta,
	       &htemp[0]+iter, 1);

    auto d_htempGPU = thrust::raw_pointer_cast(htempGPU.data());
    CudaSafeCall(cudaMemcpy(d_htempGPU, &htemp[0] + iter, iter*sizeof(real), cudaMemcpyHostToDevice));
    
    CublasSafeCall(cublasSetStream(cublas_handle, st));
    real * d_V = thrust::raw_pointer_cast(V.data());
    
    // thrust::host_vector<real> h_V = V;
    // fori(0, iter){
    //   std::cerr<<h_V[3*N*i]*z2<<" ";
    // }
    // std::cerr<<std::endl;
    // fori(0, iter){
    //   std::cerr<<h_V[3*N*i+1]*z2<<" ";
    // }
    // std::cerr<<std::endl;
    // fori(1,iter+1){
    //   forj(1,3*N)
    // 	if(h_V[3*N*i+j] != real(0.0)){
    // 	  std::cerr<<i<<" "<<j<<" "<<h_V[3*N*i+j]<<std::endl;
    // 	}
    //   if(htemp[iter+i] != real(0.0)) 	  std::cerr<<i<<" "<<htemp[iter+i]<<std::endl;
    // }
    
    /*y = ||z||_2 * Vm · H^1/2 · e1 = Vm · (z2·hdiag_temp)*/
    beta = 0.0;
    CublasSafeCall(cublasgemv(cublas_handle, CUBLAS_OP_N,
			      3*N, iter,
			      &z2,
			      d_V, 3*N,
			      d_htempGPU, 1,
			      &beta,
			      BdW, 1));
    //sys->log<System::MESSAGE>("[LanczosAlgorithm] H**1/2[0][0] = %e %e %e %e", htemp[iter], htemp[iter+1], htemp[iter+2], htemp[iter+3]);
  }

}