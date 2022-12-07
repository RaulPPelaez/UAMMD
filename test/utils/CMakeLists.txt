cmake_minimum_required(VERSION 3.14)
project(ParticleSorter)
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Release)
#set( CMAKE_VERBOSE_MAKEFILE on )
add_compile_definitions(PUBLIC MAXLOGLEVEL=5)
#GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_ARCHITECTURES OFF)

set(UAMMD_INCLUDE ../../../uammd/src ../../../uammd/src/third_party)


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

add_compile_definitions(PUBLIC DOUBLE_PRECISION)
include_directories(${UAMMD_INCLUDE})
link_libraries(${CUDA_LIBRARY} nvToolsExt)

add_executable(
  ParticleSorter
  ParticleSorter.cu
)

target_link_libraries(
  ParticleSorter
  GTest::gtest_main
  GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(ParticleSorter)


IF (CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_definitions(PUBLIC MAXLOGLEVEL=15)
  set(CMAKE_CUDA_FLAGS "-g -G  -src-in-ptx")
  set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -pedantic")
ENDIF()
