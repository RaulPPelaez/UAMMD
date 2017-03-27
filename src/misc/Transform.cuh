/*Raul P. Pelaez 2017. Transform kernel. 

  This simple template kernel allos for a general transversal of an array of the form:
     fori(0,N)
        new[i] = transform(old[i])

   This includes everything from a type change (i.e casting a real3 array to real4) to a general operation, like new = 2*old.

------------------------------------------------------------------------------------------
   The transform operation is done through a functors () operator, see an example:


   struct changetor{
   inline __device__ real3 operator()(real4 a){ return make_real3(a);}   
   };

   transform<<<N, 1>>>(old, new, changetor(), N);

 */



namespace Transform{
  template<class Told, class Tnew, class Transformer>
  __global__ void transform(Told * __restrict__ old, Tnew * __restrict__ newv, Transformer tr, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= N) return;
    newv[i] = tr(old[i]);
  }



}