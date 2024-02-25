cmake_minimum_required(VERSION 3.22)
project(examples)
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Release)

#add_compile_definitions(PUBLIC MAXLOGLEVEL=15)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

#add_compile_definitions(PUBLIC DOUBLE_PRECISION)
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

add_subdirectory(generic_md)
add_subdirectory(advanced)
add_subdirectory(basic_concepts)
add_subdirectory(integration_schemes)
add_subdirectory(integration_schemes/others)
add_subdirectory(interaction_modules)
add_subdirectory(misc)
add_subdirectory(uammd_as_a_library)


IF (CMAKE_BUILD_TYPE MATCHES "Debug")
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "-g -G  -lineinfo -src-in-ptx")
  else()
    set(CMAKE_CUDA_FLAGS "-g")
  endif()
 set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -pedantic")
ENDIF()
