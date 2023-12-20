cmake_minimum_required(VERSION 3.22)
project(tests)
enable_language(CUDA)
add_compile_definitions(PUBLIC MAXLOGLEVEL=1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

add_compile_definitions(PUBLIC DOUBLE_PRECISION)
set(UAMMD_INCLUDE ../src ../src/third_party)
set(BLA_STATIC OFF)
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
if(BLAS_FOUND)
  message("mkl environment detected")
  add_compile_definitions(PUBLIC USE_MKL)
  link_libraries(
    BLAS::BLAS
  )
  find_path(BLAS_INCLUDE_DIRS mkl.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{BLAS_HOME}/include
  )
else()
  unset(BLA_VENDOR)
  find_package(LAPACK REQUIRED)
  find_package(LAPACKE REQUIRED)
  find_package(BLAS REQUIRED)
  link_libraries(${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
  find_path(BLAS_INCLUDE_DIRS cblas.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{BLAS_HOME}/include
  )
  find_path(LAPACKE_INCLUDE_DIRS lapacke.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{LAPACKE_HOME}/include
  )

endif()
include_directories(${UAMMD_INCLUDE} ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS})
link_libraries(${CUDA_LIBRARY})
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -extended-lambda")
endif()
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(CTest)
add_subdirectory(BDHI/DPStokes)
add_subdirectory(BDHI/FCM)
add_subdirectory(BDHI/quasi2D)
add_subdirectory(utils)
add_subdirectory(misc/chebyshev)

IF (CMAKE_BUILD_TYPE MATCHES "Debug")
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "-g -G  -lineinfo -src-in-ptx")
  else()
    set(CMAKE_CUDA_FLAGS "-g")
  endif()
 set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -pedantic")
ENDIF()
