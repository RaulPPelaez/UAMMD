/*Raul P. Pelaez 2016. Vector and Matrix class, a container for CPU and GPU data, maintains both versions.


TODO:

*/
#ifndef VECTOR_H
#define VECTOR_H
#include"globals/defines.h"
#include<cstring>
#include<cuda_runtime.h>
#include"Texture.h"
#include<memory.h>
#include<algorithm>
/*Pinned memory is a little broken, it doesnt really help to make writes to disk faster, apparently pinned memory makes a CPU array accesible from the GPU, so it is not needed to download. The problem is taht writing to disk in parallel is incompatible with this, so it is better to just keep a separate CPU and GPU copies and download manually when needed*/

template<class T> class CPUVector;
template<class T> class GPUVector;
template<class T> class Vector;
template<class T> class Matrix;

#define HOSTDEVICE __host__ __device__

template<class T>
class Vector{
  int n;
  typedef T* iterator;
public:
  GPUVector<T> gpu;
  CPUVector<T> cpu;

  Vector(): gpu(), cpu(){}
  Vector(int n):n(n), gpu(n), cpu(n){}
  Vector(const GPUVector<T> &a_gpu): n(a_gpu.n), gpu(a_gpu), cpu(a_gpu){}
  Vector(const CPUVector<T> &a_cpu): n(a_cpu.n), gpu(a_cpu), cpu(a_cpu){
    std::cerr<<"From CPU"<<std::endl;}
  Vector(int n, const GPUVector<T> &a_gpu, const CPUVector<T> &a_cpu):
    n(n), gpu(a_gpu), cpu(a_cpu){}

  /*********************RULE OF FIVE*******************************/
  /*Copy Constructor*/
  Vector(const Vector<T>&other):
    Vector(other.n, other.gpu, other.cpu){}
  /*Move Constructor*/
  Vector(Vector<T>&& other) noexcept:
				   n(std::move(other.n)),				   
				   gpu(std::move(other.gpu)),
				   cpu(std::move(other.cpu)){}
  /*Copy Assignment Operator*/
  Vector<T>& operator=(const Vector<T>& other){
    Vector<T> tmp(other);
    *this = std::move(tmp);
    return *this;
  }
  /*Move Assignment Operator*/
  Vector<T>& operator=(Vector<T>&& other) noexcept{
    if(this->gpu.initialized) this->gpu.freeMem();
    if(this->cpu.h_m) this->cpu.freeMem();
    this->n = other.n;
    this->gpu = std::move(other.gpu);
    this->cpu = std::move(other.cpu);

    other.gpu.d_m = nullptr;
    other.gpu.n = 0;
    other.gpu.initialized = false;
    other.gpu.tex = 0;
    
    other.cpu.n = 0;
    other.cpu.h_m = nullptr;

    return *this;
  }
  /********************************************************************************/   

  void freeMem(){
    cpu.freeMem();
    gpu.freeMem();
  }
  
  Vector<T> & operator =(const CPUVector<T> &cpu){
    if(this->n != cpu.n)
      *this = Vector<T>(cpu);
    else{
      memcpy(this->cpu.h_m, cpu.h_m, this->n*sizeof(T));
    }
    return *this;
  }  
  Vector<T> & operator =(const GPUVector<T> &gpu){
    if(this->n != gpu.n)
      *this = Vector<T>(gpu);
    else
      gpuErrchk(cudaMemcpy(this->gpu.d_m,
			   gpu.d_m,
			   this->n*sizeof(T),
			   cudaMemcpyDeviceToDevice));      

    return *this;
  }  


  inline T*.d_m{ return this->gpu.data();}
  inline T*.data{ return this->cpu.data();}
  
  HOSTDEVICE inline T& operator [](const int &i){
#ifndef __CUDA_ARCH__
    return cpu[i];
#else
    return gpu[i];
#endif
  }


  inline void download(T* CPUdata, int start, int end){gpu.download_to(CPUdata,start, end);}
  inline void download(T* CPUdata, int end){this->download(CPUdata, 0, end);}
  inline void download(int end){this->download(cpu.data(),end);}
  inline void download(){this->download(n);}


  inline void upload(T* CPUdata, int start, int end){gpu.upload_from(CPUdata,start, end);}
  inline void upload(T* CPUdata, int end){this->upload(CPUdata, 0, end);}
  inline void upload(int end){this->upload(cpu.data(),end);}
  inline void upload(){this->upload(n);}

  
  inline operator T*&(){ return this->gpu.d_m;} /*This is just ugly, but for backwards compatibility...*/
  inline operator T*() const{ return this->gpu.data();}
  inline operator TexReference(){ return this->gpu;}
  inline operator cudaTextureObject_t(){ return this->gpu;}
  //operator shared_ptr<Vector<T>>() {return make_shared<Vector<T>>(*this);}
  
  inline TexReference getTexture(){ return this->gpu.getTexture();}

  void fill_with(T x){
    cpu.fill_with(x);
    gpu.upload_from(cpu.data());
  }


  inline iterator begin() const{ return this->cpu.h_m;}
  inline iterator end() const{ return this->cpu.h_m+n;}

  template<typename _InputIterator> 
  void assign(_InputIterator first, _InputIterator last){
    uint newSize = std::distance(first, last);
    if(newSize != n){
      n = newSize;
    }
    cpu.assign(first, last);
    gpu.upload_from(cpu.begin(), 0, newSize);
  }

  inline int size(){return this->n;}
  
  friend class Matrix<T>;
};

template<class T>
__global__ void  fill_withGPUD(T* d_m, T x, int n);

template<class T>
class GPUVector{
private:
  int n;
  bool initialized;
  T *d_m;
  cudaTextureObject_t tex;
  typedef T* iterator;
public:
  
  GPUVector(): tex(0), d_m(nullptr), n(0), initialized(false){}
  GPUVector(int N): GPUVector(){
    if(N>0){
      gpuErrchk(cudaMalloc(&d_m, N*sizeof(T)));      
      if(!d_m)std::cerr<<"Could not initialize GPUVector!!!"<<std::endl;
      else{
	this->n = N;
	this->initialized = true;
      }
    }
  }
  GPUVector(const CPUVector<T>& cpuvec):GPUVector(cpuvec.n){
    if(cpuvec.h_m){
      gpuErrchk(cudaMemcpy(d_m,
			   cpuvec.h_m,
			   this->n*sizeof(T),
			   cudaMemcpyHostToDevice));      
    }

  }
  GPUVector(T* d_m, int N): tex(0), n(N), d_m(d_m){
    if(d_m){
      initialized = true;
    }
    else{
      initialized= false;
      std::cerr<<"Invalid address in GPUVector creation!!"<<std::endl;
    }
  }
  /*********************RULE OF FIVE*******************************/
  /*Copy Constructor*/
  GPUVector(const GPUVector<T>&other):
    GPUVector(other.n){
      if(other.initialized){
	gpuErrchk(cudaMemcpy(d_m,
			     other.d_m,
			     this->n*sizeof(T),
			     cudaMemcpyDeviceToDevice));
	if(other.tex!=0) this->createTexture();
      }
  }
  /*Move Constructor*/
  GPUVector(GPUVector<T>&& other) noexcept:
				   n(std::move(other.n)),
				   initialized(std::move(other.initialized)),
				   d_m(std::move(other.d_m)),
				   tex(std::move(other.tex)){}
  /*Copy Assignment Operator*/
  GPUVector<T>& operator=(const GPUVector<T>& other){
    GPUVector<T> tmp(other);
    *this = std::move(tmp);
    return *this;
  }
  /*Move Assignment Operator*/
  GPUVector<T>& operator=(GPUVector<T>&& other) noexcept{
    if(this->initialized) this->freeMem();
    this->n = other.n;
    this->d_m = other.d_m;
    this->initialized = other.initialized;
    this->tex = other.tex;
    other.n = 0;
    other.d_m = nullptr;
    other.tex = 0;
    other.initialized = false;
    return *this;
  }
  /********************************************************************************/   
  ~GPUVector() noexcept{
    /*Using the destructor messes real bad with the CUDA enviroment when using global variables, so you have to call freeMem manually for any global Vector before main exits...*/
    this->freeMem();
    this->initialized = false;
    this->n = 0;
  }
  void freeMem(){
    if(this->initialized){
      if(d_m){gpuErrchk(cudaFree(d_m)); d_m = nullptr;}
      if(tex!=0)Texture::destroy<T>(tex);
    }

  }

  HOSTDEVICE inline TexReference getTexture(){
#ifndef __CUDA_ARCH__
    this->createTexture();
#endif
    return {(void *)d_m, tex};
  }
  inline void createTexture(){
    if(tex==0){
      tex = Texture::create<T>(d_m, n);
    }

  }

  HOSTDEVICE inline  iterator begin() const{ return this->d_m;}
  HOSTDEVICE inline iterator end() const{ return this->d_m+n;}
  HOSTDEVICE inline int size() const{return this->n;}
  HOSTDEVICE inline T*& data(){ return this->d_m;}
  inline void fill_with(T x){
    int TPB = 256;
    int nthreads = TPB<n?TPB:n;
    int nblocks  =  n/nthreads +  ((n%nthreads!=0)?1:0); 
    fill_withGPUD<<<nblocks, nthreads>>>(d_m, x, n);   
  }

  __device__ inline T texload(int i){return tex1Dfetch<T>(tex, i);}
  inline operator TexReference() const{return {(void*) d_m, tex};}
  inline operator T*() const{ return this->d_m; }
  inline operator T *&() {return d_m;}
  __device__ inline T& operator[](const int &i){ return d_m[i];}

  GPUVector<T> &operator=(const CPUVector<T> &cpu){
    if(this->n != cpu.n){
      *this = GPUVector<T>(cpu);
    }
    else{
      gpuErrchk(cudaMemcpy(d_m,
			   cpu.h_m,
			   n*sizeof(T),
			   cudaMemcpyHostToDevice));    
      
    }
    return *this;
  }  


  inline void download_to(T* CPUdata, int start, int end){
    gpuErrchk(cudaMemcpy(CPUdata,
			 d_m+start,
			 (end-start)*sizeof(T),
			 cudaMemcpyDeviceToHost));    
  }
  inline void download_to(T* CPUdata, int size){download_to(CPUdata, 0, size);} 
  inline void download_to(T* CPUdata){download_to(CPUdata, 0, n);}
  inline void upload_from(T* CPUdata, int start, int end){
    gpuErrchk(cudaMemcpy(d_m+start,
			 CPUdata,
			 (end-start)*sizeof(T),
			 cudaMemcpyHostToDevice));    
  }
  inline void upload_from(T* CPUdata, int size){upload_from(CPUdata, 0, size);}
  
  inline void upload_from(T* CPUdata){upload_from(CPUdata, 0, n);}
 

  template<typename _InputIterator> 
  void assign(_InputIterator first, _InputIterator last){
    uint newSize = std::distance(first, last);
    if(newSize != n){
      GPUVector<T> tmp(newSize);
      *this = std::move(tmp);
    }    
    gpuErrchk(cudaMemcpy(d_m, first, newSize*sizeof(T), cudaMemcpyDeviceToDevice));
  }  
  
  friend class CPUVector<T>;
  friend class Vector<T>;
};



template<class T>
class CPUVector{
  int n;
  T* h_m;
  typedef T* iterator;
public:
  CPUVector():n(0),h_m(nullptr){}
  CPUVector(int n): n(n), h_m(nullptr){
    if(n>0){
      h_m = (T *)malloc(sizeof(T)*n); //C style memory management
      if(!h_m) std::cerr<<"ERROR: Could not allocate CPUVector!!!"<<std::endl;
    }
  }
  CPUVector(const GPUVector<T> &other):CPUVector(other.n){
    if(this->n && this->h_m){
      gpuErrchk(cudaMemcpy(h_m,
			   other.d_m,
			   this->n*sizeof(T),
			   cudaMemcpyDeviceToHost));
    }
  }
  CPUVector(T* h_m, int N): n(N), h_m(h_m){
    if(!h_m){
      std::cerr<<"Invalid address in CPUVector creation!!"<<std::endl;
    }
  }
  /*********************RULE OF FIVE*******************************/
  /*Copy Constructor*/
  CPUVector(const CPUVector<T>&other):
    CPUVector(other.n){
    memcpy(h_m, other.h_m, this->n*sizeof(T));
  }
  /*Move Constructor*/
  CPUVector(CPUVector<T>&& other) noexcept:
				   n(std::move(other.n)),
                                   h_m(std::move(other.h_m)){}

  /*Copy Assignment Operator*/
  CPUVector<T>& operator=(const CPUVector<T>& other){
    CPUVector<T> tmp(other);
    *this = std::move(tmp);
    return *this;
  }
  /*Move Assignment Operator*/
  CPUVector<T>& operator=(CPUVector<T>&& other) noexcept{
    if(this->h_m) this->freeMem();
    this->n = other.n;
    this->h_m = other.h_m;
    other.n = 0;
    other.h_m = nullptr;
    return *this;
  }
  /********************************************************************************/   
  ~CPUVector() noexcept{
    /*Using the destructor messes real bad with the CUDA enviroment when using global variables, so you have to call freeMem manually for any global Vector before main exits...*/
    this->freeMem();
    this->n = 0;
  }
  inline void freeMem(){
    if(this->h_m){      
      free(h_m);
      h_m = nullptr;
    }

  }

  inline void fill_with(T x){std::fill(h_m, h_m+n, (T)x);}

  inline iterator begin(){ return this->h_m;}
  inline iterator end(){ return this->h_m+n;}

  template<typename _InputIterator> 
  void assign(_InputIterator first, _InputIterator last){
    uint newSize = std::distance(first, last);
    if(newSize != n){
      CPUVector<T> tmp(newSize);
      *this = std::move(tmp);
    }
  
    uint i = 0;
    for(auto it=first; it!=last; it++){
      this->h_m[i] = *(it);
      i++;
    }
   }

  
  inline T*& data(){return this->h_m;}
  inline int size(){return this->n;}
  inline operator T*() const{return this->h_m;}
  inline operator T*&(){return this->h_m;}
  inline T& operator [](const int &i){ return h_m[i];}

  CPUVector<T> & operator =(const GPUVector<T> &gpu){
    if(gpu.n != this->n){
      *this = CPUVector<T>(gpu);
    }
    else{
      gpuErrchk(cudaMemcpy(h_m,
			   gpu.d_m,
			   this->n*sizeof(T),
			   cudaMemcpyDeviceToHost));      
    }
    return *this;
  }
  
  friend class GPUVector<T>;
  friend class Vector<T>;
};


template<class T>
__global__ void  fill_withGPUD(T* d_m, T x, int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  d_m[i] = x;  
}



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
    for(uint i=0; i<nr; i++) M[i] = this->cpu.data() + i*nc;
  }
  /*Copy Constructor*/
  Matrix(const Matrix<T>& other) noexcept:				 
				  Vector<T>(other), nr(other.nr), nc(other.nc), M(nullptr){
    M = (T **)malloc(sizeof(T *)*nr);
    for(uint i=0; i<nr; i++) M[i] = this->cpu.data() + i*nc;
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



// //Free the CPU version of the Vector
//   void freeCPU(){
//     if(!initialized) return;
//     if(pinned){
//       cudaFreeHost(d_m);
//       d_m = nullptr;
//     }
//     else free(d_m);
//     d_m = nullptr;
//   }
//   /*Initializer Constructor*/
//   Vector(uint n, bool pinned = false):
//     n(n), pinned(pinned),
//     initialized(false), uploaded(false),
//     d_m(nullptr), d_m(nullptr), tex()
//   {
//     if(n>0){
//       this->initialized = true;
//       //Pined memory is allocated by cuda
//       if(pinned){
// 	/* cudaHostAlloc allows the array to be accesed from the CPU without copy needed, it is always updated, and there is no need for allocation on device side, the GPU address is obtained with cudaHostGetDevicePointer(&d_m, d_m, 0)
// 	 */
// 	gpuErrchk(cudaHostAlloc(&d_m, sizeof(T)*n, 0));
// 	gpuErrchk(cudaHostGetDevicePointer(&d_m, d_m, 0));
      
// 	uploaded = true;
//       }
//       else{
// 	d_m = (T *)malloc(sizeof(T)*n); //C style memory management
// 	//Allocate device memory
// 	gpuErrchk(cudaMalloc(&d_m, n*sizeof(T)));
//       }
//       if(!d_m || !d_m){
// 	std::cerr<<"Could not allocate d_m for Vector!!"<<std::endl;
// 	exit(1);
//       }
//     }
//   }
  
//   /*Copy constructor*/
//   Vector(const Vector<T>& other):
//     Vector(other.size(), other.pinned){
//     if(other.initialized){
//       std::copy(other.d_m, other.d_m+n, d_m);
//       if(other.uploaded){
// 	this->uploaded = true;
// 	gpuErrchk(cudaMemcpy(d_m, other.d_m, n*sizeof(T), cudaMemcpyDeviceToDevice));
//       }
//     }
//   }
//   /*Move constructor*/
//   Vector(Vector<T>&& other) noexcept:
// 			     n(other.n), pinned(other.pinned),
// 			     initialized(other.initialized),
// 			     uploaded(other.uploaded),
// 			     d_m(other.d_m), d_m(other.d_m)
//   {
//     other.pinned = false;
//     other.initialized = false;
//     other.n = 0;
//     other.d_m = nullptr;
//     other.d_m = nullptr;
//     other.tex.destroy();
//   }


//   /*************************************************************/

//   TexReference getTexture(){
//     if(!initialized) return {(void*)d_m, 0};
//     if(tex!=0) return {(void*) d_m,tex};
//     if(n==0 || d_m == nullptr) return {(void *)d_m, 0};
//     tex.init(d_m, n);
//     // cudaResourceDesc resDesc;
//     // memset(&resDesc, 0, sizeof(resDesc));
//     // resDesc.resType = cudaResourceTypeLinear;
//     // resDesc.res.linear.devPtr = d_m;
//     // resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
//     // resDesc.res.linear.sizeInBytes = n*sizeof(T);

//     // cudaTextureDesc texDesc;
//     // memset(&texDesc, 0, sizeof(texDesc));
//     // texDesc.readMode = cudaReadModeElementType;

//     // gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

//     return {(void*)d_m, tex};
//   }
  
//   iterator begin(){ return this->d_m;}
//   iterator end(){ return this->d_m+n;}

//   template<typename _InputIterator> 
//   void assign(_InputIterator first, _InputIterator last){
//     uint newSize = std::distance(first, last);
//     Vector<T> tmp(newSize);
//     *this = std::move(tmp);
    
//     uint i = 0;
//     for(auto it=first; it!=last; it++){
//       this->d_m[i] = *(it);
//       i++;
//     }
//   }
  
//   void fill_with(T x){std::fill(d_m, d_m+n, (T)x);}
//   //Upload/Download from the GPU, ultra fast if is pinned memory
//   inline void upload(int start, int end){
//     uploaded= true;
//     int N = end-start;
//     gpuErrchk(cudaMemcpy(d_m+start, d_m+start, N*sizeof(T), cudaMemcpyHostToDevice));
//   }
//   inline void upload(int N=0){
//     if(N==0) N=n;
//     this->upload(0,N);
//   }
  
//   inline void download(int start, int end){
//     if(!pinned){
//       int N = end-start;
//       gpuErrchk(cudaMemcpy(d_m+start, d_m+start, N*sizeof(T), cudaMemcpyDeviceToHost));
//     }
//   }
//   inline void download(int N=0){
//     if(N==0) N=n;
//     this->download(0,N);
//   }

//   inline void GPUmemset(int x){
//     uploaded= true;
//     gpuErrchk(cudaMemset(d_m, x, n*sizeof(T)));
//   }
//   inline bool GPUcopy_from(const Vector<T> &other){
//     if(other.uploaded && this->n == other.n){
//       this->uploaded = true;
//       gpuErrchk(cudaMemcpy(d_m, other.d_m, n*sizeof(T), cudaMemcpyDeviceToDevice));
//       return true;
//     }
//     return false;
//   }
//   inline bool GPUcopy_from(T* other_d_m){
//     this->uploaded = true;
//     gpuErrchk(cudaMemcpy(d_m, other_d_m, n*sizeof(T), cudaMemcpyDeviceToDevice));
//     return true;    
//   }

//   void print(){    
//     for(uint i=0; i<n; i++)
//       cout<<d_m[i]<<" ";
//     cout<<std::endl;
    
//   }

//   uint size() const{return this->n;}
//   //void operator =(const Vector<T> &a){fori(0,n) d_m[i] = a[i]; }
//   //Access d_m with bracket operator
//   T& operator [](const int &i){return d_m[i];}
//   //Cast to float* returns the device pointer!
//   operator T *&() {return d_m;}
//   operator T *() const{return d_m;}
//   operator shared_ptr<Vector<T>>() {return make_shared<Vector<T>>(*this);}
//   operator cudaTextureObject_t(){ return this->getTexture().tex; }
// };

// template<class T>
// class Matrix: public Vector<T>{
// public:
//   uint nr, nc;
//   T **M;//Pointers to each column
//   Matrix(): Vector<T>(0),
//     nr(0), nc(0),
//     M(nullptr){}
//   Matrix(uint nr, uint nc): Vector<T>(nr*nc),
//     nr(nr),nc(nc),
//     M(nullptr){
//     M = (T **)malloc(sizeof(T *)*nr);
//     for(uint i=0; i<nr; i++) M[i] = this->d_m + i*nc;
//   }
//   /*Copy Constructor*/
//   Matrix(const Matrix<T>& other) noexcept:				 
// 				  Vector<T>(other), nr(other.nr), nc(other.nc), M(nullptr){
//     M = (T **)malloc(sizeof(T *)*nr);
//     for(uint i=0; i<nr; i++) M[i] = this->d_m + i*nc;
//   }
//   //TODO This one doesnt work
//   /*Move constructor*/
//   Matrix(Matrix<T>&& other) noexcept:
// 			     Vector<T>(std::move(other)),
//     nr(other.nr), nc(other.nc),
//     M(std::move(other.M)){
    
//     other.nc = 0;
//     other.nr = 0;
//     other.M = nullptr;
//   }
//   /*Copy assignement operator*/
//   Matrix<T>& operator=(const Matrix<T>& other){
//     Matrix<T> tmp(other);
//     *this = std::move(tmp);
//     return *this;
//   }
//   /*Move assignement operator*/
//   Matrix<T>& operator= (Matrix<T>&& other) noexcept{
//     Vector<T>::operator=(std::move(other));

//     this->nc = other.nc;
//     this->nr = other.nr;
//     this->M = other.M;

//     other.nc = 0;
//     other.nr = 0;
//     other.M = nullptr;
    
//     return *this;

//   }

    
//   ~Matrix() noexcept{
//     if(M) free(M);
//     nr = 0;
//     nc = 0;
//     M = nullptr;
//   }
//   int2 size() const{return make_int2(this->nr, this->nc);}
//   bool isSquare(){ return nr==nc;}
//   bool isSym(){
//     if(!this->isSquare()) return false;
//     fori(0,nc-1){
//       forj(i+1, nc)
// 	if(M[i][j] != M[j][i]) return false;
//     }
//     return true;
//   }
//   void print(){
//     fori(0,(int)nr){
//       forj(0, (int)nc) cout<<M[i][j]<<"\t";
//       cout<<std::endl;
//     }
//   }
//   T*& operator [](const int &i){return M[i];}
// };

#endif
