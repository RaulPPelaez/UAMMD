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
typedef unsigned int uint;

template<class T>
class Vector{
  typedef T* iterator;
public:
  uint n; //size of the matrix
  bool pinned; //Flag to use pinned memory
  bool initialized;
  bool uploaded;
  T *data; //The data itself, stored aligned in memory
  T *d_m; //device pointer

  //  cudaTextureObject_t tex;
  Texture<T> tex;
  //Free the CPU version of the Vector
  void freeCPU(){
    if(!initialized) return;
    if(pinned){
      cudaFreeHost(data);
      d_m = nullptr;
    }
    else free(data);
    data = nullptr;
  }
  //Free the device memory
  void freeGPU(){
    if(this->initialized){
      if(!pinned && d_m){gpuErrchk(cudaFree(d_m)); d_m = nullptr;}
      if(tex!=0)tex.destroy();
    }

  }
  void freeMem(){
    freeCPU();
    freeGPU();
    n = 0;
    initialized = false;
    pinned = false;
    uploaded = false;
  }

  /********RULE OF FIVE******************/
  
  /*Default constructor*/
  Vector():n(0), pinned(false),
	   initialized(false),uploaded(false),
	   data(nullptr), d_m(nullptr), tex(0){}
  /*Destructor*/
  ~Vector() noexcept{
    /*Using the destructor messes real bad with the CUDA enviroment when using global variables, so you have to call freeMem manually for any global Vector before main exits...*/
    freeMem();
  }
  /*Initializer Constructor*/
  Vector(uint n, bool pinned = false):
    n(n), pinned(pinned),
    initialized(false), uploaded(false),
    data(nullptr), d_m(nullptr), tex(0)
  {
    if(n>0){
      this->initialized = true;
      //Pined memory is allocated by cuda
      if(pinned){
	/* cudaHostAlloc allows the array to be accesed from the CPU without copy needed, it is always updated, and there is no need for allocation on device side, the GPU address is obtained with cudaHostGetDevicePointer(&d_m, data, 0)
	 */
	gpuErrchk(cudaHostAlloc(&data, sizeof(T)*n, 0));
	gpuErrchk(cudaHostGetDevicePointer(&d_m, data, 0));
      
	uploaded = true;
      }
      else{
	data = (T *)malloc(sizeof(T)*n); //C style memory management
	//Allocate device memory
	gpuErrchk(cudaMalloc(&d_m, n*sizeof(T)));
      }
      if(!data || !d_m){
	cerr<<"Could not allocate data for Vector!!"<<endl;
	exit(1);
      }
    }
  }
  
  /*Copy constructor*/
  Vector(const Vector<T>& other):
    Vector(other.size(), other.pinned){
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
			     n(other.n), pinned(other.pinned),
			     initialized(other.initialized),
			     uploaded(other.uploaded),
			     data(other.data), d_m(other.d_m)
  {
    other.pinned = false;
    other.initialized = false;
    other.n = 0;
    other.data = nullptr;
    other.d_m = nullptr;
    other.tex.destroy();
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
    this->pinned = other.pinned;
    this->uploaded = other.uploaded;
    this->tex.destroy();
    other.n = 0;
    other.data = nullptr;
    other.d_m = nullptr;
    other.initialized = false;
    other.uploaded = false;
    other.pinned = false;
    other.tex.destroy();
    return *this;
  }

  /*************************************************************/

  cudaTextureObject_t getTexture(){
    if(!initialized) return 0;
    if(tex!=0) return tex;
    if(n==0 || d_m == nullptr) return 0;
    tex.init(d_m, n);
    // cudaResourceDesc resDesc;
    // memset(&resDesc, 0, sizeof(resDesc));
    // resDesc.resType = cudaResourceTypeLinear;
    // resDesc.res.linear.devPtr = d_m;
    // resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
    // resDesc.res.linear.sizeInBytes = n*sizeof(T);

    // cudaTextureDesc texDesc;
    // memset(&texDesc, 0, sizeof(texDesc));
    // texDesc.readMode = cudaReadModeElementType;

    // gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    return tex;
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
  inline void upload(){
    uploaded= true;
    gpuErrchk(cudaMemcpy(d_m, data, n*sizeof(T), cudaMemcpyHostToDevice));
  }
  inline void download(){
    if(!pinned)
      gpuErrchk(cudaMemcpy(data, d_m, n*sizeof(T), cudaMemcpyDeviceToHost));
  }

  inline void GPUmemset(T x){
    uploaded= true;
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
  void print(){
    download();
    for(int i=0; i<n; i++)
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
  operator shared_ptr<Vector<T>>() {return make_shared<Vector<T>>(*this);}
  operator cudaTextureObject_t(){ return this->getTexture(); }
};

template<class T>
class Matrix: public Vector<T>{
public:
  uint nr, nc;
  T **M;//Pointers to each column
  Matrix(): Vector<T>(0),
    nr(0), nc(0),
    M(nullptr){}
  Matrix(uint nc, uint nr): Vector<T>(nr*nc),
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
    fori(0,nr){
      forj(0, nc) cout<<M[i][j]<<"\t";
      cout<<endl;
    }
  }
  T*& operator [](const int &i){return M[i];}
};

#endif
