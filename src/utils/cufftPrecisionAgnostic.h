/*Raul P. Pelaez 2018. cufft precision abstractions
  The names in this file allow to use cuFFT in a precision agnostic way.

  Instead of using cufftDoubleReal and cufftReal use cufftReal_t<double> or cufftReal_t<float>.
  And so on for several cuFFT functions

*/
#ifndef CUFFTPRECISIONAGNOSTIC_H
#define CUFFTPRECISIONAGNOSTIC_H
#include<cufft.h>
namespace uammd{
  //Transforming cufft macros to templates...
  namespace detail{
    template<class T> struct cufftTypeAgnostic;
    template<> struct cufftTypeAgnostic<double>{using type=cufftDoubleReal;};
    template<> struct cufftTypeAgnostic<float>{using type=cufftReal;};

    template<class T> struct cufftComplexType;
    template<> struct cufftComplexType<double>{using type=cufftDoubleComplex;};
    template<> struct cufftComplexType<float>{using type=cufftComplex;};
  }

  template<class T> using cufftReal_t = typename detail::cufftTypeAgnostic<T>::type;
  template<class T> using cufftComplex_t = typename detail::cufftComplexType<T>::type;

  template<class T> struct CUFFT_Real2Complex;
  template<> struct CUFFT_Real2Complex<double>{static constexpr cufftType value=CUFFT_D2Z;};
  template<> struct CUFFT_Real2Complex<float>{static constexpr cufftType value=CUFFT_R2C;};

  template<class prec>
  inline
  cufftResult cufftExecReal2Complex(cufftHandle &plan, cufftReal_t<prec>* d_in, cufftComplex_t<prec> *d_out);

  template<>
  inline
  cufftResult cufftExecReal2Complex<float>(cufftHandle &plan, cufftReal_t<float>* d_in, cufftComplex_t<float> *d_out){
    return cufftExecR2C(plan, d_in, d_out);
  }

  template<>
  inline
  cufftResult cufftExecReal2Complex<double>(cufftHandle &plan, cufftReal_t<double>* d_in, cufftComplex_t<double> *d_out){
    return cufftExecD2Z(plan, d_in, d_out);
  }

  template<class T> struct CUFFT_Complex2Real;
  template<> struct CUFFT_Complex2Real<double>{static constexpr cufftType value=CUFFT_Z2D;};
  template<> struct CUFFT_Complex2Real<float>{static constexpr cufftType value=CUFFT_C2R;};

  template<class real>
  inline
  cufftResult cufftExecComplex2Real(cufftHandle &plan, cufftComplex_t<real>* d_in, cufftReal_t<real> *d_out);

  template<>
  inline
  cufftResult cufftExecComplex2Real<float>(cufftHandle &plan, cufftComplex_t<float> *d_in, cufftReal_t<float> *d_out){
    return cufftExecC2R(plan, d_in, d_out);
  }

  template<>
  inline
  cufftResult cufftExecComplex2Real<double>(cufftHandle &plan, cufftComplex_t<double> *d_in, cufftReal_t<double> *d_out){
    return cufftExecZ2D(plan, d_in, d_out);
  }

  template<class T> struct CUFFT_Complex2Complex;
  template<> struct CUFFT_Complex2Complex<double>{static constexpr cufftType value=CUFFT_Z2Z;};
  template<> struct CUFFT_Complex2Complex<float>{static constexpr cufftType value=CUFFT_C2C;};

  template<class real>
  inline
  cufftResult cufftExecComplex2Complex(cufftHandle &plan, cufftComplex_t<real>* d_in, cufftComplex_t<real> *d_out, int direction);

  template<>
  inline
  cufftResult cufftExecComplex2Complex<float>(cufftHandle &plan, cufftComplex_t<float> *d_in, cufftComplex_t<float> *d_out, int direction){
    return cufftExecC2C(plan, d_in, d_out, direction);
  }

  template<>
  inline
  cufftResult cufftExecComplex2Complex<double>(cufftHandle &plan, cufftComplex_t<double> *d_in, cufftComplex_t<double> *d_out, int direction){
    return cufftExecZ2Z(plan, d_in, d_out, direction);
  }

}
#endif
