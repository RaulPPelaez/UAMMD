/*Raul P. Pelaez 2016. Vector and Matrix class, a container for CPU and GPU data, maintains both versions.


TODO:

*/
#ifndef VECTOR_H
#define VECTOR_H
#include"globals/defines.h"
#include<cstring>
#include<cuda_runtime.h>
#include "Texture.h"
#include<memory.h>
#include<algorithm>
/*Pinned memory is a little broken, it doesnt really help to make writes to disk faster, apparently pinned memory makes a CPU array accesible from the GPU, so it is not needed to download. The problem is taht writing to disk in parallel is incompatible with this, so it is better to just keep a separate CPU and GPU copies and download manually when needed*/
template<class T>
class Vector{
  typedef T* iterator;
public:
  uint n; //size of the matrix
  bool initialized;
  bool uploaded;
  T *data; //The data itself, stored aligned in memory
  T *d_m; //device pointer

  cudaTextureObject_t tex;
  //Texture<T> tex;
  //Free the CPU version of the Vector
  void freeCPU(){
    if(!initialized) return;
    free(data);
    data = nullptr;
  }
  //Free the device memory
  void freeGPU(){
    if(this->initialized){
      if(d_m){gpuErrchk(cudaFree(d_m)); d_m = nullptr;}
      if(tex!=0) Texture::destroy(tex);
    }

  }
  void freeMem(){
    freeCPU();
    freeGPU();
    n = 0;
    initialized = false;
    uploaded = false;
  }

  /********RULE OF FIVE******************/
  
  /*Default constructor*/
  Vector():n(0),
	   initialized(false),uploaded(false),
	   data(nullptr), d_m(nullptr), tex(0){}
  /*Destructor*/
  ~Vector() noexcept{
    /*Using the destructor messes real bad with the CUDA enviroment when using global variables, so you have to call freeMem manually for any global Vector before main exits...*/
    freeMem();
  }
  /*Initializer Constructor*/
  Vector(uint n):
    n(n),
    initialized(false), uploaded(false),
    data(nullptr), d_m(nullptr), tex(0)
  {
    if(n>0){
      this->initialized = true;     
      data = (T *)malloc(sizeof(T)*n); //C style memory management
      //Allocate device memory
      gpuErrchk(cudaMalloc(&d_m, n*sizeof(T)));
      
      if(!data || !d_m){
	cerr<<"Could not allocate data for Vector!!"<<endl;
	exit(1);
      }
    }
  }
  
  /*Copy constructor*/
  Vector(const Vector<T>& other):
    Vector(other.size()){
    if(other.initialized){
      std::copy(other.data, other.data+n, data);
      if(other.uploaded){
	this->uploaded = true;
	gpuErrchk(cudaMemcpy(d_m, other.d_m, n*sizeof(T), cudaMemcpyDeviceToDevice));
      }
    }
  }
  /*Move constructor*/
  Vector(Vector<T>&& other) noexcept:
			     n(other.n),
			     initialized(other.initialized),
			     uploaded(other.uploaded),
			     data(other.data), d_m(other.d_m), tex(other.tex)
  {
    other.initialized = false;
    other.n = 0;
    other.data = nullptr;
    other.d_m = nullptr;
    other.tex = 0;
  }

  /*Copy assignement operator*/
  Vector<T>& operator=(const Vector<T>& other){
    Vector<T> tmp(other);
    *this = std::move(tmp);
    return *this;
  }
  /*Move assignement operator*/
  Vector<T>& operator= (Vector<T>&& other) noexcept{
    if(initialized){
      this->freeMem();
    }
    this->n = other.size();
    this->data = other.data;
    this->d_m = other.d_m;
    this->initialized = other.initialized;
    this->uploaded = other.uploaded;
    this->tex = other.tex;
    other.n = 0;
    other.data = nullptr;
    other.d_m = nullptr;
    other.initialized = false;
    other.uploaded = false;
    other.tex = 0;
    return *this;
  }

  /*************************************************************/

  TexReference getTexture(){
    if(!initialized) return {(void*)d_m, 0};
    if(tex!=0) return {(void*) d_m,tex};
    if(n==0 || d_m == nullptr) return {(void *)d_m, 0};
    Texture::init<T>(d_m, tex, n);
    return {(void*)d_m, tex};
  }
  
  iterator begin(){ return this->data;}
  iterator end(){ return this->data+n;}

  template<typename _InputIterator> 
  void assign(_InputIterator first, _InputIterator last){
    uint newSize = std::distance(first, last);
    Vector<T> tmp(newSize);
    *this = std::move(tmp);
    
    uint i = 0;
    for(auto it=first; it!=last; it++){
      this->data[i] = *(it);
      i++;
    }
  }
  
  void fill_with(T x){std::fill(data, data+n, (T)x);}
  //Upload/Download from the GPU, ultra fast if is pinned memory
  inline void upload(int start, int end){
    uploaded= true;
    int N = end-start;
    gpuErrchk(cudaMemcpy(d_m+start, data+start, N*sizeof(T), cudaMemcpyHostToDevice));
  }
  inline void upload(int N=0){
    if(N==0) N=n;
    this->upload(0,N);
  }
  
  inline void download(int start, int end){
    int N = end-start;
    gpuErrchk(cudaMemcpy(data+start, d_m+start, N*sizeof(T), cudaMemcpyDeviceToHost));
  }
  inline void download(int N=0){
    if(N==0) N=n;
    this->download(0,N);
  }

  inline void GPUmemset(uint x){
    uploaded= true;
    gpuErrchk(cudaMemset(d_m, x, n*sizeof(T)));
  }
  inline void memset(uint x){
    uploaded= true;
    std::memset(data, x, n*sizeof(T));
    gpuErrchk(cudaMemset(d_m, x, n*sizeof(T)));
  }

  inline bool GPUcopy_from(const Vector<T> &other){
    if(other.uploaded && this->n == other.n){
      this->uploaded = true;
      gpuErrchk(cudaMemcpy(d_m, other.d_m, n*sizeof(T), cudaMemcpyDeviceToDevice));
      return true;
    }
    return false;
  }
  inline bool GPUcopy_from(T* other_d_m){
    this->uploaded = true;
    gpuErrchk(cudaMemcpy(d_m, other_d_m, n*sizeof(T), cudaMemcpyDeviceToDevice));
    return true;    
  }

  void print(){    
    for(uint i=0; i<n; i++)
      cout<<data[i]<<" ";
    cout<<endl;
    
  }

  uint size() const{return this->n;}
  //void operator =(const Vector<T> &a){fori(0,n) data[i] = a[i]; }
  //Access data with bracket operator
  T& operator [](const int &i){return data[i];}
  //Cast to float* returns the device pointer!
  operator T *&() {return d_m;}
  operator T *() const{return d_m;}
  operator cudaTextureObject_t(){ return this->getTexture().tex; }
};

template<class T>
__global__ void GPUfill_with(T* d_m, T x, int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  d_m[i] = x;
}

template<class T>
class GPUVector{
  typedef T* iterator;
public:
  uint n; //size of the matrix
  bool initialized;
  T *d_m; //device pointer
  //Texture<T> tex;
  cudaTextureObject_t tex;
  //Free the device memory
  void freeMem(){
    if(this->initialized){
      if(d_m){gpuErrchk(cudaFree(d_m)); d_m = nullptr;}
      if(tex!=0)Texture::destroy(tex);
      initialized = false;
      n = 0;
    }

  }

  /********RULE OF FIVE******************/
  
  /*Default constructor*/
  GPUVector():n(0),
	      initialized(false),
	      d_m(nullptr), tex(0){}
  /*Destructor*/
  ~GPUVector() noexcept{
    /*Using the destructor messes real bad with the CUDA enviroment when using global variables, so you have to call freeMem manually for any global Vector before main exits...*/
    freeMem();
  }
  /*Initializer Constructor*/
  GPUVector(uint n):
    n(n),
    initialized(false),
    d_m(nullptr), tex(0)
  {
    if(n>0){
      this->initialized = true;
      //Allocate device memory
      gpuErrchk(cudaMalloc(&d_m, n*sizeof(T)));
      if(!d_m){
	cerr<<"Could not allocate data for GPUVector!!"<<endl;
	exit(1);
      }
    }
  }
  
  /*Copy constructor*/
  GPUVector(const GPUVector<T>& other):
    GPUVector(other.size()){
    if(other.initialized)
      gpuErrchk(cudaMemcpy(d_m, other.d_m, n*sizeof(T), cudaMemcpyDeviceToDevice));
  }
  /*Move constructor*/
  GPUVector(GPUVector<T>&& other) noexcept:
				   n(other.n),
				   initialized(other.initialized),
				   d_m(other.d_m), tex(other.tex)
  {
    other.initialized = false;
    other.d_m = nullptr;
    other.tex = 0;
  }
  
  /*Copy assignement operator*/
  GPUVector<T>& operator=(const GPUVector<T>& other){
    GPUVector<T> tmp(other);
    *this = std::move(tmp);
    return *this;
  }
  /*Move assignement operator*/
  GPUVector<T>& operator= (GPUVector<T>&& other) noexcept{
    if(initialized){
      this->freeMem();
    }
    this->n = other.size();
    this->d_m = other.d_m;
    this->initialized = other.initialized;
    this->tex = other.tex;
    other.n = 0;
    other.d_m = nullptr;
    other.initialized = false;
    other.tex = 0;
    return *this;
  }

  /*************************************************************/

  TexReference getTexture(){
    if(!initialized) return {(void*)d_m, 0};
    if(tex!=0) return {(void*) d_m,tex};
    if(n==0 || d_m == nullptr) return {(void *)d_m, 0};
    Texture::init<T>(d_m, tex, n);
    return {(void*)d_m, tex};
  }
  
  iterator begin(){ return this->d_m;}
  iterator end(){ return this->d_m+n;}
  
  //Upload/Download from the GPU, ultra fast if is pinned memory
  inline void upload(T* data, int start, int end){
    int N = end-start;
    gpuErrchk(cudaMemcpy(d_m+start, data+start, N*sizeof(T), cudaMemcpyHostToDevice));
  }
  inline void upload(T* data, int N=0){
    if(N==0) N=n;
    this->upload(data, 0,N);
  }
  
  inline void download(T* data, int start, int end){
    int N = end-start;
    gpuErrchk(cudaMemcpy(data+start, d_m+start, N*sizeof(T), cudaMemcpyDeviceToHost));
  }
  inline void download(T* data, int N=0){
    if(N==0) N=n;
    this->download(data, 0,N);
  }

  inline void memset(int x, cudaStream_t st = 0){
    if(st != 0){
      gpuErrchk(cudaMemsetAsync(d_m, x, n*sizeof(T), st));
    }
    else{
      gpuErrchk(cudaMemset(d_m, x, n*sizeof(T)));
    }
  }
  inline bool copy_fromGPU(T* other_d_m){
    this->uploaded = true;
    gpuErrchk(cudaMemcpy(d_m, other_d_m, n*sizeof(T), cudaMemcpyDeviceToDevice));
    return true;    
  }
  inline void fill_with(T x){
    int BLOCKSIZE = 1024;
    int nthreads = BLOCKSIZE<n?BLOCKSIZE:n;
    int nblocks  =  n/nthreads +  ((n%nthreads!=0)?1:0); 
    GPUfill_with<T><<<nblocks, nthreads>>>(d_m, x, n);    
  }
  uint size() const{return this->n;}
  //void operator =(const Vector<T> &a){fori(0,n) data[i] = a[i]; }
  //Access data with bracket operator
  inline __device__ T& operator [](const int &i){return d_m[i];}
  //Cast to float* returns the device pointer!
  operator T *&() {return d_m;}
  operator T *() const{return d_m;}
  operator shared_ptr<GPUVector<T>>() {return make_shared<GPUVector<T>>(*this);}
  operator cudaTextureObject_t(){ return this->getTexture().tex; }
};










template<class T>
class Matrix: public Vector<T>{
public:
  uint nr, nc;
  T **M;//Pointers to each column
  Matrix(): Vector<T>(0),
    nr(0), nc(0),
    M(nullptr){}
  Matrix(uint nr, uint nc): Vector<T>(nr*nc),
    nr(nr),nc(nc),
    M(nullptr){
    M = (T **)malloc(sizeof(T *)*nr);
    for(uint i=0; i<nr; i++) M[i] = this->data + i*nc;
  }
  /*Copy Constructor*/
  Matrix(const Matrix<T>& other) noexcept:				 
				  Vector<T>(other), nr(other.nr), nc(other.nc), M(nullptr){
    M = (T **)malloc(sizeof(T *)*nr);
    for(uint i=0; i<nr; i++) M[i] = this->data + i*nc;
  }
  //TODO This one doesnt work
  /*Move constructor*/
  Matrix(Matrix<T>&& other) noexcept:
			     Vector<T>(std::move(other)),
    nr(other.nr), nc(other.nc),
    M(std::move(other.M)){
    
    other.nc = 0;
    other.nr = 0;
    other.M = nullptr;
  }
  /*Copy assignement operator*/
  Matrix<T>& operator=(const Matrix<T>& other){
    Matrix<T> tmp(other);
    *this = std::move(tmp);
    return *this;
  }
  /*Move assignement operator*/
  Matrix<T>& operator= (Matrix<T>&& other) noexcept{
    Vector<T>::operator=(std::move(other));

    this->nc = other.nc;
    this->nr = other.nr;
    this->M = other.M;

    other.nc = 0;
    other.nr = 0;
    other.M = nullptr;
    
    return *this;

  }

    
  ~Matrix() noexcept{
    if(M) free(M);
    nr = 0;
    nc = 0;
    M = nullptr;
  }
  int2 size() const{return make_int2(this->nr, this->nc);}
  bool isSquare(){ return nr==nc;}
  bool isSym(){
    if(!this->isSquare()) return false;
    fori(0,nc-1){
      forj(i+1, nc)
	if(M[i][j] != M[j][i]) return false;
    }
    return true;
  }
  void print(){
    fori(0,(int)nr){
      forj(0, (int)nc) cout<<M[i][j]<<"\t";
      cout<<endl;
    }
  }
  T*& operator [](const int &i){return M[i];}
};

#endif
