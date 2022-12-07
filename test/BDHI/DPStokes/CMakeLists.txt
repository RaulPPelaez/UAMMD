cmake_minimum_required(VERSION 3.14)
project(dpstokes)
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Debug)
#set( CMAKE_VERBOSE_MAKEFILE on )
#add_compile_definitions(PUBLIC MAXLOGLEVEL=15)
#GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_ARCHITECTURES OFF)

set(UAMMD_INCLUDE ../../../../uammd/src ../../../../uammd/src/third_party)


include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()



set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
if(BLAS_FOUND)
  message("mkl environment detected")
  add_compile_definitions(PUBLIC USE_MKL)
else()
  unset(BLA_VENDOR)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(LAPACKE REQUIRED)
endif()

add_compile_definitions(PUBLIC DOUBLE_PRECISION)
include_directories(${UAMMD_INCLUDE})
link_libraries(${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES} ${BLAS_LIBRARIES} ${CUDA_LIBRARY} cufft cublas cusolver)

add_executable(
  dpstokes_test
  dpstokes_test.cu
)

target_link_libraries(
  dpstokes_test
  GTest::gtest_main
  GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(dpstokes_test)

IF (CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_options(PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -G -src-in-ptx
    >)
ENDIF()
