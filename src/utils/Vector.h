/*Raul P. Pelaez 2016. Vector and Matrix class, a container for CPU and GPU data, maintains both versions.


TODO:
100- Look for a way to make the clean up automatically without using the destructor.
100-Fix the rule of five for Matrix
*/
#ifndef VECTOR_H
#define VECTOR_H


/*Pinned memory doesnt seem to help*/
template<class T>
class Vector{
public:
  T *data; //The data itself, stored aligned in memory
  T *d_m; //device pointer
  uint n; //size of the matrix
  bool pinned; //Flag to use pinned memory
  bool initialized;
  bool uploaded;
  //Free the CPU version of the Vector
  void freeCPU(){
    if(pinned) cudaFreeHost(data);
    else free(data);
  }
  //Free the device memory
  void freeGPU(){ if(!pinned)gpuErrchk(cudaFree(d_m));}
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
  Vector():initialized(false),uploaded(false), pinned(false),
	   n(0), data(nullptr), d_m(nullptr){}
  /*Destructor*/
  ~Vector() noexcept{
    /*Using the destructor messes real bad with the CUDA enviroment when using global variables, so
      it is better to not use them here and delete manually...*/
    //freeCPU();
    //freeGPU();
  }
  /*Initializer Constructor*/
  Vector(uint n, bool pinned = false):
    n(n), pinned(pinned),
    initialized(true), uploaded(false),
    data(nullptr), d_m(nullptr)
  {
    //Pined memory is allocated by cuda
    if(pinned){
      gpuErrchk(cudaHostAlloc(&data, sizeof(T)*n, cudaHostAllocMapped));
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
      freeCPU();
      freeGPU();
    }
    this->n = other.size();
    this->data = other.data;
    this->d_m = other.d_m;
    this->initialized = other.initialized;
    this->pinned = other.pinned;
    this->uploaded = other.uploaded;
    other.n = 0;
    other.data = nullptr;
    other.d_m = nullptr;
    other.initialized = false;
    other.uploaded = false;
    other.pinned = false;
    
    return *this;
  }

  /*************************************************************/
  void fill_with(T x){std::fill(data, data+n, (T)x);}
  //Upload/Download from the GPU, ultra fast if is pinned memory
  inline void upload(){
    uploaded= true;
    gpuErrchk(cudaMemcpy(d_m, data, n*sizeof(T), cudaMemcpyHostToDevice));
  }
  inline void download(){
    gpuErrchk(cudaMemcpy(data, d_m, n*sizeof(T), cudaMemcpyDeviceToHost));
  }

  void print() const{
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
};

typedef Vector<float4> Vector4;
typedef Vector<float3> Vector3;
#define Vector4Ptr shared_ptr<Vector4> 
#define Vector3Ptr shared_ptr<Vector3> 

template<class T>
class Matrix: public Vector<T>{
public:
  T **M;//Pointers to each column
  uint nr, nc;
  Matrix(uint n, uint m): nr(n),nc(m), Vector<T>(nr*nc){
    M = (T **)malloc(sizeof(T *)*nr);
    for(int i=0; i<nr; i++) M[i] = &(this->data)[i*nc];
  }
  ~Matrix() noexcept{
    free(M);
  }
  int2 size() const{return make_int2(this->nr, this->nc);}
  T*& operator [](const int &i){return M[i];}
};

#endif
