#ifndef LANCZOS_MATRIX_DOT_H
#define LANCZOS_MATRIX_DOT_H
#include "global/defines.h"
#include <functional>
namespace uammd {
namespace lanczos {
struct MatrixDot {
  void setSize(int newsize) { this->m_size = newsize; }
  // virtual void dot(real* v, real*Mv) = 0;
  virtual void operator()(real *v, real *Mv) = 0;

protected:
  int m_size;
};

// Transforms any callable into a MatrixDot valid to use with Lanczos
template <class Foo> struct MatrixDotAdaptor : public lanczos::MatrixDot {
  Foo &foo;
  MatrixDotAdaptor(Foo &&foo) : foo(foo) {}
  void operator()(real *v, real *Mv) override { foo(v, Mv); }
};

template <class Foo> auto createMatrixDotAdaptor(Foo &&foo) {
  return MatrixDotAdaptor<Foo>(foo);
}

} // namespace lanczos
} // namespace uammd
#endif
