
#include"LanczosAlgorithm.cuh"
#include <nvToolsExt.h>
#include<lapacke.h>
#include<fstream>

void LanczosAlgorithm::init(){
  /*Init cuSolver and cuBLAS for Lanczos process*/
  status = cublasCreate(&cublas_handle);
  if(status){cerr<<"ERROR with CUBLAS!!\n"<<endl; exit(1);}
  cusolverDnCreate(&solver_handle);
  h_work_size = 0;//work size of operation  
  cusolverDngesvd_bufferSize(solver_handle, max_iter, max_iter, &h_work_size);

  gpuErrchk(cudaMalloc(&d_work, h_work_size));
  gpuErrchk(cudaMalloc(&d_info, sizeof(int)));

  /*Create noise*/
  curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(curng, grng.next());
  /*Create a temporal vector to warm up curand*/
  GPUVector3 noise(N);
  //Curand fill with gaussian numbers with mean 0 and var 1
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(curng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  
}

LanczosAlgorithm::LanczosAlgorithm(int N, int max_iter):
  N(N),
  h_work_size(0), d_work(nullptr), d_info(nullptr),
  max_iter(max_iter),  
  w(N),
  V(3*N*max_iter),
  H(max_iter*max_iter), Htemp(max_iter*max_iter),
  P(max_iter*max_iter), Pt(max_iter*max_iter),
  hdiag(max_iter), hsup(max_iter), hdiag_temp(max_iter),
  old_noise(N)
{
  P.fill_with(0);
}


void LanczosAlgorithm::compNoise(real z2, int N, int iter, real * BdW){

  real alpha = 1.0;
  real beta = 0.0;
   /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/
   /**** H^1/2·e1 = Pt· first_column_of(sqrt(Hdiag)·P) ******/
 


   // auto info = LAPACKE_ssteqr(LAPACK_COL_MAJOR, 'i',
   // 			     iter, hdiag.data, hsup.data,
   // 			      P.data, iter);

   // cerr<<info<<endl;

   // /***Hdiag_temp = Hdiag·P·e1****/
   // forj(0,iter){
   //   hdiag_temp[j] = sqrt(hdiag[j])*P[N*j];
   // }
   // if(info<0)
   //   fori(0,iter*iter) if(P[i]!=0) cerr<<i<<endl;
   
   // hdiag_temp.upload();  
   // P.upload();
   // /***** Htemp = H^1/2·e1 = Pt· hdiag_temp ****/
   // cublasgemv(cublas_handle,
   //    	     CUBLAS_OP_T,
   //    	     iter, iter,
   //    	     &alpha,
   //    	     P.d_m, iter,
   //    	     hdiag_temp.d_m, iter,
   //    	     &beta,
   //   	     Htemp.d_m, iter);
   // beta = 0;
   // cublasStatus_t st = cublasgemv(cublas_handle, CUBLAS_OP_N,
   // 				 3*N, iter,
   // 				 &z2,
   // 				 V, 3*N,
   // 				 Htemp.d_m, 1,
   // 				 &beta,
   // 				 BdW, 1);


  fori(0,iter) hdiag_temp[i] = hdiag[i];
  /*Compute H^1/2 by diagonalization***********/
  /* H^1/2 = P · Hdiag^1/2 · P^T */
  /******WITH CUSOLVER SVD****************/
   fori(0,iter*iter) H.data[i] = 0;
   fori(0,iter){
     H.data[i*iter+i] = hdiag_temp[i];
     if((uint)i<iter-1){
       H.data[(i+1)*iter+i] = hsup[i];
     }
     if(i>0){
       H.data[(i-1)*iter+i] = hsup[i-1];
     }
   }
   //  nvtxRangePushA("CusolverGesv");  
   H.upload(iter*iter);
   /*Diagonalize H*/
   cusolverDngesvd(solver_handle, 'A', 'A', iter, iter,
   		  H.d_m, iter,
   		  hdiag_temp.d_m,
   		  P.d_m, iter,
   		  Pt.d_m, iter,
   		  d_work, h_work_size, nullptr ,d_info);  

   //nvtxRangePop();
   hdiag_temp.download(iter);
   
   // fori(0,max_iter){
   //   cerr<<hdiag[i]<<endl;
   // }
   // exit(1);
   
   /*sqrt H*/
   fori(0, iter){
     if(hdiag_temp[i]>0)
       hdiag_temp[i] = sqrt(hdiag_temp[i]);
     else hdiag_temp[i] = 0.0;
   }    
   /* store in hdiag_temp -> H^1/2·e1 = P·Hdiag_Temp^1/2·Pt·e1*/
   fori(0,iter*iter) H.data[i] = 0;
   fori(0,iter) H.data[i*iter+i] = hdiag_temp[i];
  
   H.upload(iter*iter);

   cublasgemm(cublas_handle,
     	     CUBLAS_OP_N, CUBLAS_OP_N,
     	     iter,iter,iter,
     	     &alpha,
     	     H, iter,
     	     Pt, iter,
     	     &beta,
     	     Htemp,  iter);
   cublasgemm(cublas_handle,
     	     CUBLAS_OP_N,  CUBLAS_OP_N,
     	     iter, iter, iter,
     	     &alpha,
     	     P, iter,
     	     Htemp, iter,
     	     &beta,
    	     H, iter);
   /*Now H contains H^1/2*/  
   /*Store H^1/2 · e1 in hdiag_temp*/
   /*hdiag_temp = H^1/2·e1*/

   cudaMemcpy(hdiag_temp.d_m, H.d_m, iter*sizeof(real), cudaMemcpyDeviceToDevice);
  
  /*noise = ||z||_2 * Vm · H^1/2 · e1 = Vm · (z2·hdiag_temp)*/
  beta = 0.0;
   cublasStatus_t st = cublasgemv(cublas_handle, CUBLAS_OP_N,
   				 3*N, iter,
   				 &z2,
   				 V, 3*N,
   				 hdiag_temp.d_m, 1,
   				 &beta,
   				 BdW, 1);
  
  /*Print BdW*/
  
   // Vector3 tmp(N);
   // tmp.upload();
   // cudaMemcpy(tmp.d_m, BdW, N*sizeof(real3), cudaMemcpyDeviceToDevice);
   // tmp.download();
   // ofstream out("noiseLanc.dat");
   // fori(0,N)
   //   out<<tmp[i]<<endl;
   // out.close();
   // exit(0);
}

