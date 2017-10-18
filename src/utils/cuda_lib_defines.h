/*Raul P. Pelaez 2016. Precision agnostic cublas/cusolver function defines*/
#include"global/defines.h"


#if defined SINGLE_PRECISION
#define cusolverDnpotrf cusolverDnSpotrf
#define cusolverDnpotrf_bufferSize cusolverDnSpotrf_bufferSize
#define cublastrmv cublasStrmv
#define cublassymv cublasSsymv
#define cublasgemv cublasSgemv
#define cublasnrm2 cublasSnrm2
#define cublasscal cublasSscal
#define cublasaxpy cublasSaxpy
#define cublasdot cublasSdot
#define cusolverDnsyevd cusolverDnSsyevd
#define cusolverDnsyevd_bufferSize cusolverDnSsyevd_bufferSize
#define cusolverDngesvd_bufferSize cusolverDnSgesvd_bufferSize
#define cusolverDngesvd cusolverDnSgesvd
#define cublasgemm cublasSgemm
#else
#define cusolverDnpotrf cusolverDnDpotrf
#define cusolverDnpotrf_bufferSize cusolverDnDpotrf_bufferSize
#define cublastrmv cublasDtrmv
#define curandGenerateNormal curandGenerateNormalDouble
#define cublassymv cublasDsymv
#define cublasgemv cublasDgemv
#define cublasnrm2 cublasDnrm2
#define cublasscal cublasDscal
#define cublasaxpy cublasDaxpy
#define cublasdot cublasDdot
#define cusolverDnsyevd cusolverDnDsyevd
#define cusolverDnsyevd_bufferSize cusolverDnDsyevd_bufferSize
#define cusolverDngesvd_bufferSize cusolverDnDgesvd_bufferSize
#define cusolverDngesvd cusolverDnDgesvd
#define cublasgemm cublasDgemm
#endif


