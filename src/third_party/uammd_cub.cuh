#pragma once

#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include"cub_bak/cub/cub.cuh"
#endif
