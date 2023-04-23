#pragma once

//Detect CUDA compilation using clang
#if defined(__clang__) && defined(__CUDA__) ||  __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include"cub_bak/cub/cub.cuh"
#endif
