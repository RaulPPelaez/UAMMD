/* P. Palacios Alonso 2021
Some quaternion algebra and useful functions
Notation:
   n(real)  - Scalar part 
   v(real3) - vectorial part
   q(real4) - quaternion
 */

#ifndef QUATERNION_CUH
#define QUATERNION_CUH

#define QUATTR inline __host__ __device__

namespace uammd{


  class Quat{
  public:
    
    real n;
    real3 v;
    
    QUATTR Quat();
    QUATTR Quat(real4 q);
    QUATTR Quat(real n, real3 v);
    QUATTR Quat(real n, real vx, real vy, real vz);

    QUATTR Quat operator+(Quat q1);
    QUATTR void operator+=(Quat q);
    QUATTR Quat operator-(Quat q);
    QUATTR void operator-=(Quat q);
    QUATTR Quat operator*(Quat q);
    QUATTR void operator*=(Quat q);
    QUATTR Quat operator*(real scalar);
    QUATTR void operator*=(real scalar);
    QUATTR Quat operator/(real scalar);
    QUATTR void operator/=(real scalar);
    QUATTR void operator=(real4 q);
    
    QUATTR real3 getVx(); //Returns the first vector of the reference system encoded by the quaternion
    QUATTR real3 getVy(); //Returns the second vector of the reference system encoded by the quaternion
    QUATTR real3 getVz(); //Returns the third vector of the reference system encoded by the quaternion
      QUATTR real4 to_real4();
  };
  
  QUATTR Quat::Quat(){
    n = real(1.0);
    v = real3();
  }
  
  QUATTR Quat::Quat(real4 q){
    n = q.x;
    v.x = q.y;
    v.y = q.z;
    v.z = q.w;
  }
  
  QUATTR Quat::Quat(real n, real3 v){
    this->n = n;
    this->v = v;
  }
  
  QUATTR Quat::Quat(real n, real vx, real vy, real vz){
    this -> n = n;
    v.x = vx;
    v.y = vy;
    v.z = vz;	
  }
  
  QUATTR Quat Quat::operator+(Quat q){
    return Quat(n+q.n,v+q.v);    
  }

  QUATTR void Quat::operator+=(Quat q){
    n+=q.n;
    v+=q.v;
  }
  
  QUATTR Quat Quat::operator-(Quat q){
    return Quat(n-q.n,v-q.v);    
    }
  
  QUATTR void Quat::operator-=(Quat q){
    n-=q.n;
    v-=q.v;
  }
  
  QUATTR Quat Quat::operator*(Quat q){
    /*
      Product of two quaternions:
      q3 = q1*q2 = (n1*n2 - v1*v2, n1*v2 + n2*v1 + v1 x v2)
    */      
    return Quat(n*q.n-dot(v,q.v),n*q.v+v*q.n+cross(v,q.v));    
  }
  
  QUATTR void Quat::operator*=(Quat q){
    n=n*q.n-dot(v,q.v);
    v=n*q.v+v*q.n+cross(q.v,v);
  }
  
  QUATTR Quat Quat::operator*(real scalar){
    return Quat(scalar*n,scalar*v);    
  }
  
  QUATTR void Quat::operator*=(real scalar){
      n*=scalar;
      v*=scalar;
  }
  
  QUATTR Quat operator*(real scalar, Quat q){
    return  Quat(scalar*q.n,scalar*q.v);    
  }
  
  QUATTR Quat Quat::operator/(real scalar){
    return Quat(n/scalar,v/scalar);    
  }
  
  QUATTR void Quat::operator/=(real scalar){
    n/=scalar;
    v/=scalar;
  }
  
  QUATTR void Quat::operator=(real4 q){
    n = q.x;
    v.x = q.y;
    v.y = q.z;
    v.z = q.w;
  }

  QUATTR real3 Quat::getVx(){
    real a = n;
    real b = v.x;
    real c = v.y;
    real d = v.z;
    real3 vx = make_real3(a*a+b*b-c*c-d*d,2*(b*c+a*d),2*(b*d-a*c));
    return vx;
  }
  
  QUATTR real3 Quat::getVy(){
    real a = n;
    real b = v.x;
    real c = v.y;
    real d = v.z;
    return make_real3(2*(b*c-a*d),a*a-b*b+c*c-d*d,2*(c*d+a*b));
  }
  
  QUATTR real3 Quat::getVz(){
    real a = n;
    real b = v.x;
    real c = v.y;
    real d = v.z;
    return make_real3(2*(b*d+a*c),2*(c*d-a*b),a*a-b*b-c*c+d*d);  
  }
  
  QUATTR real4 Quat::to_real4(){
    real4 r4;
    r4.x = n;
    r4.y = v.x;
    r4.z = v.y;
    r4.w = v.z;
    return r4;
  }
  
  QUATTR Quat rotVec2Quaternion(real3 vrot, real phi){
    /* Returns the quaternion that encondes a rotation of ang radians 
       around the axis vrot 	 
       q = (cos(phi/2),vrot·sin(phi/2))
    */
    vrot*=rsqrt(dot(vrot,vrot)); // The rotation axis must be a unitary vector
    real cphi2, sphi2;
    real* cphi2_ptr = &cphi2;
    real* sphi2_ptr = &sphi2;
    sincos(phi*0.5,sphi2_ptr,cphi2_ptr);
    Quat q = Quat(cphi2,sphi2*vrot);
    return q;
  }
  
  QUATTR Quat rotVec2Quaternion(real3 vrot){
    // If no angle is given the rotation angle is the modulus of vrot
    real phi = sqrt(dot(vrot,vrot));
    return rotVec2Quaternion(vrot,phi);
  }
}

#endif
