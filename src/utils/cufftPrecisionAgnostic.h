/*Raul P. Pelaez 2018. cufft precision abstractions
  The names in this file allow to use cuFFT in a precision agnostic way.

  Instead of using cufftDoubleReal and cufftReal use cufftReal_t<double> or cufftReal_t<double>.
  And so on for several cuFFT functions

*/
#ifndef CUFFTPRECISIONAGNOSTIC_H
#define CUFFTPRECISIONAGNOSTIC_H
#include<cufft.h>
namespace uammd{
  //Transforming cufft macros to templates...
  template<class T> struct cufftTypeAgnostic;
  template<> struct cufftTypeAgnostic<double>{using type=cufftDoubleReal;};
  template<> struct cufftTypeAgnostic<float>{using type=cufftReal;};
  template<class T> using cufftReal_t = typename cufftTypeAgnostic<T>::type;

  template<class T> struct cufftComplexType;
  template<> struct cufftComplexType<double>{using type=cufftDoubleComplex;};
  template<> struct cufftComplexType<float>{using type=cufftComplex;};
  template<class T> using cufftComplex_t = typename cufftComplexType<T>::type;

  template<class T> struct CUFFT_Real2Complex;
  template<> struct CUFFT_Real2Complex<double>{static constexpr cufftType value=CUFFT_D2Z;};
  template<> struct CUFFT_Real2Complex<float>{static constexpr cufftType value=CUFFT_R2C;};

  template<class real>
  void cufftExecReal2Complex(cufftHandle &plan,
			     cufftReal_t<real>* d_in,
			     cufftComplex_t<real> *d_out);
  template<>
  void cufftExecReal2Complex<float>(cufftHandle &plan,
				    cufftReal_t<float>* d_in,
				    cufftComplex_t<float> *d_out){

    cufftExecR2C(plan, d_in, d_out);
  }

  template<>
  void cufftExecReal2Complex<double>(cufftHandle &plan,
				     cufftReal_t<double>* d_in,
				     cufftComplex_t<double> *d_out){

    cufftExecD2Z(plan, d_in, d_out);
  
  }


  template<class T> struct CUFFT_Complex2Real;
  template<> struct CUFFT_Complex2Real<double>{static constexpr cufftType value=CUFFT_Z2D;};
  template<> struct CUFFT_Complex2Real<float>{static constexpr cufftType value=CUFFT_C2R;};

  template<class real>
  void cufftExecComplex2Real(cufftHandle &plan,
			     cufftComplex_t<real>* d_in,
			     cufftReal_t<real> *d_out);
  template<>
  void cufftExecComplex2Real<float>(cufftHandle &plan,
				    cufftComplex_t<float> *d_in,
				    cufftReal_t<float> *d_out){
    cufftExecC2R(plan, d_in, d_out);

  }

  template<>
  void cufftExecComplex2Real<double>(cufftHandle &plan,
				     cufftComplex_t<double> *d_in,
				     cufftReal_t<double> *d_out){

    cufftExecZ2D(plan, d_in, d_out);
  }
}
#endif
