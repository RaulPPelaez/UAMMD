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

namespace uammd {
class Quat {
public:
  real n;
  real3 v;

  QUATTR Quat();
  QUATTR Quat(const real4 &q);
  QUATTR Quat(real n, const real3 &v);
  QUATTR Quat(real n, real vx, real vy, real vz);

  QUATTR Quat operator+(const Quat &q) const;
  QUATTR void operator+=(const Quat &q);
  QUATTR Quat operator-(const Quat &q) const;
  QUATTR void operator-=(const Quat &q);
  QUATTR Quat operator*(const Quat &q) const;
  QUATTR void operator*=(const Quat &q);
  QUATTR Quat operator*(real scalar) const;
  QUATTR void operator*=(real scalar);
  QUATTR Quat operator/(real scalar) const;
  QUATTR void operator/=(real scalar);
  QUATTR void operator=(const real4 &q);

  QUATTR real3 getVx() const;
  QUATTR real3 getVy() const;
  QUATTR real3 getVz() const;
  QUATTR real4 to_real4() const;

  QUATTR Quat getConjugate() const;
};

QUATTR Quat::Quat() {
  n = real(1.0);
  v = real3();
}

QUATTR Quat::Quat(const real4 &q) {
  n = q.x;
  v.x = q.y;
  v.y = q.z;
  v.z = q.w;
}

QUATTR Quat::Quat(real n, const real3 &v) {
  this->n = n;
  this->v = v;
}

QUATTR Quat::Quat(real n, real vx, real vy, real vz) {
  this->n = n;
  v.x = vx;
  v.y = vy;
  v.z = vz;
}

QUATTR Quat Quat::operator+(const Quat &q) const {
  return Quat(n + q.n, v + q.v);
}

QUATTR void Quat::operator+=(const Quat &q) {
  n += q.n;
  v += q.v;
}

QUATTR Quat Quat::operator-(const Quat &q) const {
  return Quat(n - q.n, v - q.v);
}

QUATTR void Quat::operator-=(const Quat &q) {
  n -= q.n;
  v -= q.v;
}

QUATTR Quat Quat::operator*(const Quat &q) const {
  /*
     Product of two quaternions:
     q3 = q1*q2 = (n1*n2 - v1*v2, n1*v2 + n2*v1 + v1 x v2)
   */
  return Quat(n * q.n - dot(v, q.v), n * q.v + v * q.n + cross(v, q.v));
}

QUATTR void Quat::operator*=(const Quat &q) {
  real n_new = n * q.n - dot(v, q.v);
  real3 v_new = n * q.v + v * q.n + cross(v, q.v);

  n = n_new;
  v = v_new;
}

QUATTR Quat Quat::operator*(real scalar) const {
  return Quat(scalar * n, scalar * v);
}

QUATTR void Quat::operator*=(real scalar) {
  n *= scalar;
  v *= scalar;
}

QUATTR Quat operator*(real scalar, const Quat &q) {
  return Quat(scalar * q.n, scalar * q.v);
}

QUATTR Quat Quat::operator/(real scalar) const {
  return Quat(n / scalar, v / scalar);
}

QUATTR void Quat::operator/=(real scalar) {
  n /= scalar;
  v /= scalar;
}

QUATTR void Quat::operator=(const real4 &q) {
  n = q.x;
  v.x = q.y;
  v.y = q.z;
  v.z = q.w;
}

QUATTR real3 Quat::getVx() const {
  real a = n;
  real b = v.x;
  real c = v.y;
  real d = v.z;
  real3 vx =
      make_real3(a * a + b * b - c * c - d * d, real(2.0) * (b * c + a * d),
                 real(2.0) * (b * d - a * c));
  return vx;
}

QUATTR real3 Quat::getVy() const {
  real a = n;
  real b = v.x;
  real c = v.y;
  real d = v.z;
  return make_real3(real(2.0) * (b * c - a * d), a * a - b * b + c * c - d * d,
                    real(2.0) * (c * d + a * b));
}

QUATTR real3 Quat::getVz() const {
  real a = n;
  real b = v.x;
  real c = v.y;
  real d = v.z;
  return make_real3(real(2.0) * (b * d + a * c), real(2.0) * (c * d - a * b),
                    a * a - b * b - c * c + d * d);
}

QUATTR real4 Quat::to_real4() const {
  real4 r4;
  r4.x = n;
  r4.y = v.x;
  r4.z = v.y;
  r4.w = v.z;
  return r4;
}

QUATTR real4 make_real4(Quat q) { return q.to_real4(); }

QUATTR Quat Quat::getConjugate() const { return Quat(n, -v); }

/*
   Returns the quaternion that encondes a rotation of ang radians
   around the axis vrot
   q = (cos(phi/2),vrot·sin(phi/2))
 */
QUATTR Quat rotVec2Quaternion(real3 vrot, real phi) {
  real norm = (dot(vrot, vrot));
  if (norm == real(0.0))
    return Quat(1.0, 0., 0., 0.);
  vrot *= rsqrt(norm); // The rotation axis must be a unitary vector
  real cphi2, sphi2;
  real *cphi2_ptr = &cphi2;
  real *sphi2_ptr = &sphi2;
  sincos(phi * real(0.5), sphi2_ptr, cphi2_ptr);
  Quat q = Quat(cphi2, sphi2 * vrot);
  return q;
}

QUATTR Quat rotVec2Quaternion(real3 vrot) {
  // If no angle is given the rotation angle is the modulus of vrot
  real phi = sqrt(dot(vrot, vrot));
  return rotVec2Quaternion(vrot, phi);
}

/* Rotates a vector v, with the rotation encoded by the quaternion q, the
   formula of the rotation is p' = q*p*q^1, being p a quaternion of the form
   [0,v]. To speed up the computation, we write the rotation in the next form:
   v' = v + 2*q_n*(q_v x v) + 2*q_v * (q_v x v).
   See
   https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
 */
QUATTR real3 rotateVector(const Quat &q, const real3 &v) {
  real3 aux = real(2.0) * cross(q.v, v);
  return v + q.n * aux + cross(q.v, aux);
}
} // namespace uammd

#endif
