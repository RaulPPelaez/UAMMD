
#define NLOW 300

#include"Diagonalize.h"

#include<vector>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<cmath>
#include<iostream>
using namespace std;

void eigenVecCPU(const real *a,const real *b,const real *eig, real *Q, int N);
/*Use 2.6.12. cublas<t>tbsv() in this one*/
void eigenVecGPU(real *a, real *b, real *eig, real *Q, int N);
void QRsymtridiag(real *a, real *b,  real *anext, real *bnext, real *P, real*Q, int N);


void triDiagEigCPU(real *hdiag, real *hsup, real *P, uint N){

  /*Temporal storage*/
  static vector<real> anext(N,0), bnext(N,0);
  static vector<real> a(N,0), b(N,0), Q(N*N,0);

  if(anext.size()<N){
    anext.resize(N,0);
    bnext.resize(N,0);
    a.resize(N,0);
    b.resize(N,0);
    Q.resize(N*N,0);
  }
  fori(0,N*N) Q[i] = 0;
  fori(0,N){
    a[i] = hdiag[i];
    b[i] = hsup[i];
    Q[i*N+i] = 1;
  }
  
  bool converged = false;
  while(!converged){
    /*Perform 10 QR decompositions, switching between the two array pairs*/
    fori(0,5){
      QRsymtridiag(a.data(), b.data(), anext.data(), bnext.data(),Q.data(), P, N);
      QRsymtridiag(anext.data(), bnext.data(), a.data(), b.data(),P, Q.data(), N);
    }
    /*Check for convergence, using the Gershgorin circle theorem*/
    fori(0,N-1){
      if(b[i]>1e-5) break;
      if(i==N-2) converged = true;
    }    
  }

  fori(0,N){
    hdiag[i] = a[i];
    hsup[i]  = b[i];
  }
  
  if(P==nullptr) return;
  fori(0,N*N) P[i] = Q[i];
  // /*Solve the tridiagonal lineal system*/
  // eigenVecCPU(hdiag, hsup, a.data(), P, N);
}

void triDiagEigGPU(real *a, real *b, real *P, uint N){
  /*TODO*/
  triDiagEigCPU(a,b,P,N);
    
}

/*On exit, hdiag contains the eigenvalues, 
           hsup should be epsilon ~ 0
       and P contains the eigenvectors*/
void triDiagEig(real *hdiag, real *hsup, real *P, uint N){
  if(N<NLOW){
    /*Do in CPU*/
    triDiagEigCPU(hdiag,hsup,P,N);
  }
  else{
    /*Do in GPU*/
    triDiagEigGPU(hdiag,hsup,P,N);

  }

}


/*Reference: The Computer Journal (1963) 6 (1): 99-101. doi: 10.1093/comjnl/6.1.99 */

/*Performs a QR decomposition
  on a tridiagonal symmetric matrix, given its diagonal and superior diagonal*/
/*Returns the new matrix in anext, bnext*/
void QRsymtridiag(real *a, real *b,  real *anext, real *bnext, real* P,  real* Q, int N){


  real u = 0;
  real s2 = 0, s, c;

  real p2=0, l=0;

  real btil2 = 0;
  real b2;
  real a_iplus1 = 0;

  memset(anext, 0, N*sizeof(real));
  memset(bnext, 0, N*sizeof(real));
  
  //fori(0,N) anext[i] = bnext[i] = 0.0;
  
  for(int i=0; i<N; i++){
    
    l = a[i] - u;

    if(s2!=1.0f) p2 = l*l/(1.0f-s2);
    else         p2 = (1.0f-s2)*b[i]*b[i];
    
    if(i<N-1) b2 = b[i]*b[i];
    else b2 = 0;
    
    btil2 = s2*(p2+b2);
    s2 = b2/(p2+b2);
    if(i<N-1) a_iplus1 = a[i+1];
    else a_iplus1 = 0;
    u = s2*(l+ a_iplus1);

    anext[i] = l+u;
    if(i>0)bnext[i-1] = sqrt(btil2);

    s = sqrt(s2);
    c = sqrt(p2/(p2+b2));
    if(i<N-1)
      forj(0,N){
	Q[j*N+i]   =  c*P[j*N+i]+s*P[j*N+i+1];
	Q[j*N+i+1] = -s*P[j*N+i]+c*P[j*N+i+1];
	

      }
    
  }
  bnext[N-1] = 0;

  
  
}




/*Solve the tridiagonal symmetric linear system to get the eigenvectors*/
/*Needs the diagonal and upper part of the original matrix and its eigenvalues*/
/*Q is an NxN matrix with the eigenvectors in each column*/
void eigenVecCPU(const real *a,const  real *b, const real *eig, real *Q, int N){
  /*Solving the system using the Tridiagonal matrix algorithm
    https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm */

  // static vector<real> c(N,0);

  // if(c.size()<N){
  //   c.resize(N,0);
  // }
  // /*For each autovector*/
  // forj(0,N){
  //   /*Forward sweep*/
  //   c[0] = b[0]/a[0];

  //   fori(1,N){      
  //     c[i] = b[i]/(a[i]-b[i-1]*c[i-1]);
  //   }
    
  //   /*Backwards substitution*/
    
  //   Q[j*N+N-1] =1;
  //   real norm = Q[j*N+N-1]*Q[j*N+N-1];
  //   for(int i=N-2; i>=0; i--){
  //     Q[j*N+i] = -c[i]*Q[j*N+i+1];
  //     norm += Q[j*N+i]*Q[j*N+i];
  //   }
  //    norm = sqrt(norm);
  //    fori(0,N){
  //      //Q[j*N+i] /= norm;
  //    }
  // }
  
  /*******/
   real norm = 0.0;
   for(int i=0; i<N; i++){ /*Autovector i*/
     Q[i*N+(N-1)] = 1.0;
     Q[i*N+(N-2)] = -Q[i*N+(N-1)]*(a[N-1]-eig[i])/b[N-2];
     for(int j=N-2; j>1; j--){
       Q[i*N+(j-1)] = -(Q[i*N+j]*(a[j]-eig[i])  + Q[i*N+(j+1)]*b[j])/b[j-1];
     }
     Q[i*N+0] = -Q[i*N+1]*b[0]/(a[0]-eig[i]);
  
     norm = 0.0;
     for(int j=N-1; j>=0 ; j--){
       norm += Q[i*N+j]*Q[i*N+j];
     }
     
     real sqnorm = sqrt(norm);
     for(int j=0; j<N ; j++){
       cerr<<Q[i*N+j]<<" ";
       Q[i*N+j] /= sqnorm;
     }
     cerr<<endl;

   }


}    
