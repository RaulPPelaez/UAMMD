/*Raul P. Pelaez 2016. Precision agnostic cublas/cusolver function defines*/
#ifndef CUDA_LIB_DEFINES_H
#define CUDA_LIB_DEFINES_H
#include"global/defines.h"
#include<cublas_v2.h>

#if defined SINGLE_PRECISION
#define cusolverDnpotrf cusolverDnSpotrf
#define cusolverDnpotrf_bufferSize cusolverDnSpotrf_bufferSize
#define cublastrmv cublasStrmv
#define curandgeneratenormal curandGenerateNormal
#define cublassymv cublasSsymv
#define cublasgemv cublasSgemv
#define cublasnrm2 cublasSnrm2
#define cublasscal cublasSscal
#define cublasaxpy cublasSaxpy
#define cublasdot cublasSdot
#define cusolverDnsyevd cusolverDnSsyevd
#define cusolverDnsyevd_bufferSize cusolverDnSsyevd_bufferSize
#define cusolverDngesvd_bufferSize cusolverDnSgesvd_bufferSize
#define cusolverDngetrf_bufferSize cusolverDnSgetrf_bufferSize
#define cusolverDncgetrf_bufferSize cusolverDnCgetrf_bufferSize
#define cusolverDngesvd cusolverDnSgesvd
#define cusolverDngetrf cusolverDnSgetrf
#define cusolverDngetrs cusolverDnSgetrs
#define cusolverDncgetrf cusolverDnCgetrf
#define cusolverDncgetrs cusolverDnCgetrs
#define cublasgemm cublasSgemm
#define cublasrgemv cublasSgemv
#define cublascgemm cublasCgemm
#define cublascgemv cublasCgemv
#else
#define cusolverDnpotrf cusolverDnDpotrf
#define cusolverDnpotrf_bufferSize cusolverDnDpotrf_bufferSize
#define cublastrmv cublasDtrmv
#define curandgeneratenormal curandGenerateNormalDouble
#define cublassymv cublasDsymv
#define cublasgemv cublasDgemv
#define cublasnrm2 cublasDnrm2
#define cublasscal cublasDscal
#define cublasaxpy cublasDaxpy
#define cublasdot cublasDdot
#define cusolverDnsyevd cusolverDnDsyevd
#define cusolverDnsyevd_bufferSize cusolverDnDsyevd_bufferSize
#define cusolverDngesvd_bufferSize cusolverDnDgesvd_bufferSize
#define cusolverDngetrf_bufferSize cusolverDnDgetrf_bufferSize
#define cusolverDncgetrf_bufferSize cusolverDnZgetrf_bufferSize
#define cusolverDngesvd cusolverDnDgesvd
#define cusolverDngetrf cusolverDnDgetrf
#define cusolverDngetrs cusolverDnDgetrs
#define cusolverDncgetrf cusolverDnZgetrf
#define cusolverDncgetrs cusolverDnZgetrs
#define cublasgemm cublasDgemm
#define cublasrgemv cublasDgemv
#define cublascgemm cublasZgemm
#define cublascgemv cublasZgemv
#endif


#endif
