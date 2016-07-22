/*Raul P. Pelaez 2016. Vector and Matrix class, a container for CPU and GPU data, maintains both versions.


TODO:
100- Look for a way to make the clean up automatically without using the destructor.
100-Fix the rule of five for Matrix
*/
#ifndef VECTOR_H
#define VECTOR_H




/*Pinned memory is a little broken, it doesnt really help to make writes to disk faster, apparently pinned memory makes a CPU array accesible from the GPU, so it is not needed to download. The problem is taht writing to disk in parallel is incompatible with this, so it is better to just keep a separate CPU and GPU copies and download manually when needed*/
template<class T>
class Vector{
  typedef T* iterator;
public:
  T *data; //The data itself, stored aligned in memory
  T *d_m; //device pointer
  uint n; //size of the matrix
  bool pinned; //Flag to use pinned memory
  bool initialized;
  bool uploaded;
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
    if(this->initialized)
      if(!pinned && d_m){gpuErrchk(cudaFree(d_m)); d_m = nullptr;}
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
  Vector():initialized(false),uploaded(false), pinned(false),
	   n(0), data(nullptr), d_m(nullptr){}
  /*Destructor*/
  ~Vector() noexcept{
    /*Using the destructor messes real bad with the CUDA enviroment when using global variables, so you have to call freeMem manually for any global Vector...*/
    freeMem();
  }
  /*Initializer Constructor*/
  Vector(uint n, bool pinned = false):
    n(n), pinned(pinned),
    initialized(false), uploaded(false),
    data(nullptr), d_m(nullptr)
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
    other.n = 0;
    other.data = nullptr;
    other.d_m = nullptr;
    other.initialized = false;
    other.uploaded = false;
    other.pinned = false;
    
    return *this;
  }

  /*************************************************************/
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
  Matrix(uint nc, uint nr): nr(nr),nc(nc), Vector<T>(nr*nc){
    M = (T **)malloc(sizeof(T *)*nr);
    for(int i=0; i<nr; i++) M[i] = this->data + i*nc;
  }
  ~Matrix() noexcept{
    if(this->initialized)
      free(M);
    nr = 0;
    nc = 0;
  }
  int2 size() const{return make_int2(this->nr, this->nc);}
  T*& operator [](const int &i){return M[i];}
};


#endif
