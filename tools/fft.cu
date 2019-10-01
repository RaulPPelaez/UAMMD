/* Raul P. Pelaez 2017. Fast Fourier Transform

   Takes a single column signal and outputs its FFT (using cuda) in frequency domain.

Usage:

fft [N] [Fs] < signal

N: Number of points in the signal
Fs: Sampling frequency

Example:

seq 0 0.2 10000 |  awk '{print sin($1)}' | fft 10000 5.0 > kk

w=$(grep $( datamash -W max 2 <kk) kk | awk '{print 2*3.1415*$1}')

w will be 1

 */
#include<iostream>
#include<cufft.h>
#include<vector>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/extrema.h>

#include<cmath>
#include<string>
#include<cstdlib>

inline __host__ __device__ double mod(const cufftComplex &v){
  return sqrt(v.x*v.x + v.y*v.y);
}
int main(int argc, char *argv[]){
  if(argc<3){
    std::cerr<<"ERROR!!: Input missing"<<std::endl;
    std::cerr<<"Takes a single column signal and outputs its FFT (using cuda) in frequency domain."<<std::endl;
    std::cerr<<""<<std::endl;
    std::cerr<<"Usage:"<<std::endl;
    std::cerr<<""<<std::endl;
    std::cerr<<"fft [N] [Fs] < signal"<<std::endl;
    std::cerr<<""<<std::endl;
    std::cerr<<"N: Number of points in the signal"<<std::endl;
    std::cerr<<"Fs: Sampling frequency"<<std::endl;
    std::cerr<<""<<std::endl;
    std::cerr<<"Example:"<<std::endl;
    std::cerr<<""<<std::endl;
    std::cerr<<"seq 0 0.2 10000 |  awk '{print sin($1)}' | fft 10000 5.0 > kk"<<std::endl;
    std::cerr<<""<<std::endl;
    std::cerr<<"w=$(grep $( datamash -W max 2 <kk) kk | awk '{print 2*3.1415*$1}')"<<std::endl;
    std::cerr<<std::endl;
    std::cerr<<"w will be 1"<<std::endl;
    exit(1);
  }


  int numberElements = std::atoi(argv[1]);
  double Fs = std::stod(argv[2]);

  cufftHandle plan;
  thrust::device_vector<cufftComplex> data(numberElements/2+1);

  thrust::host_vector<cufftComplex> h_data(numberElements/2+1, cufftComplex());

  cufftReal *h_in = (cufftReal*) thrust::raw_pointer_cast(h_data.data());
  for(int i = 0; i<numberElements; i++){
    std::cin>>h_in[i];
  }
  //Upload
  data = h_data;

  //Create and execute cuFFT plan
  cufftPlan1d(&plan, numberElements, CUFFT_R2C, 1);

  cufftComplex *d_m = thrust::raw_pointer_cast(data.data());

  cufftExecR2C(plan, (cufftReal*) d_m, d_m);

  cudaDeviceSynchronize();

  //Download
  h_data = data;


  thrust::device_vector<double> devAmplitudes(numberElements/2+1);

  //Print
  thrust::transform(data.begin(),
		    data.begin()+numberElements/2+1,
		    devAmplitudes.begin(),
		    [=] __device__ (cufftComplex a){ return 2.0*mod(a)/(double)numberElements;}
		    );

  double maxAmplitude = *thrust::max_element(devAmplitudes.begin(),
					     devAmplitudes.end());

  thrust::device_vector<double> hostAmplitudes = devAmplitudes;


  for(int i = 0; i<numberElements/2+1; i++){

    double Aw = hostAmplitudes[i];
    double phase = 0.0;
    if(Aw >= maxAmplitude/1000.0)
      phase = std::atan2(h_data[i].y, h_data[i].x);


    std::cout<<i*Fs/(numberElements)<<" "<<Aw<<" "<<phase<<"\n";
  }
  std::cout<<std::flush;

  cudaDeviceSynchronize();
  return 0;
}