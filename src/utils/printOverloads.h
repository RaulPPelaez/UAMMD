/*Raul P. Pelaez 2022. Overloads to print/read uammd vector types
 */

#include "utils/tensor.cuh"
#include "utils/vector.cuh"
#include <istream>
#include <ostream>

inline std::ostream &operator<<(std::ostream &out, const uammd::real2 &f) {
  return out << f.x << " " << f.y;
}

inline std::ostream &operator<<(std::ostream &out, const uammd::real3 &f) {
  return out << f.x << " " << f.y << " " << f.z;
}

inline std::ostream &operator<<(std::ostream &out, const uammd::real4 &f) {
  return out << f.x << " " << f.y << " " << f.z << " " << f.w;
}

inline std::ostream &operator<<(std::ostream &out, const int3 &f) {
  return out << f.x << " " << f.y << " " << f.z;
}

inline std::ostream &operator<<(std::ostream &out, const int4 &f) {
  return out << f.x << " " << f.y << " " << f.z << " " << f.w;
}

inline std::istream &operator>>(std::istream &in, float2 &f) {
  return in >> f.x >> f.y;
}

inline std::istream &operator>>(std::istream &in, float3 &f) {
  return in >> f.x >> f.y >> f.z;
}

inline std::istream &operator>>(std::istream &in, float4 &f) {
  return in >> f.x >> f.y >> f.z >> f.w;
}

inline std::istream &operator>>(std::istream &in, double2 &f) {
  return in >> f.x >> f.y;
}

inline std::istream &operator>>(std::istream &in, double3 &f) {
  return in >> f.x >> f.y >> f.z;
}

inline std::istream &operator>>(std::istream &in, double4 &f) {
  return in >> f.x >> f.y >> f.z >> f.w;
}

inline std::istream &operator>>(std::istream &in, int3 &f) {
  return in >> f.x >> f.y >> f.z;
}

inline std::istream &operator>>(std::istream &in, int4 &f) {
  return in >> f.x >> f.y >> f.z >> f.w;
}

inline std::ostream &operator<<(std::ostream &os, const uammd::tensor3 &t) {
  os << t.xx << " " << t.xy << " " << t.xz << " " << t.yx << " " << t.yy << " "
     << t.yz << " " << t.zx << " " << t.zy << " " << t.zz;
  return os;
}
