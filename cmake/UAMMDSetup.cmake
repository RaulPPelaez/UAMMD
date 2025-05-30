include_guard(GLOBAL)

function(uammd_setup_target target_name)
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA is required but not enabled! \
        You must call enable_language(CUDA) in your main CMakeLists.txt BEFORE calling uammd_setup_target().")
  endif()
  if (NOT TARGET BLAS::BLAS)
    find_package(BLAS REQUIRED)
  endif()
  if (NOT TARGET LAPACK::LAPACK)
    find_package(LAPACK REQUIRED)
  endif()
  # Ensure the target has C++ and CUDA standards
  target_compile_features(${target_name} PRIVATE cxx_std_14)
  target_compile_features(${target_name} PRIVATE cuda_std_14)

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(${target_name} PRIVATE --expt-relaxed-constexpr -extended-lambda)
  endif()
  target_link_libraries(${target_name} PRIVATE BLAS::BLAS LAPACK::LAPACK)
  set(BLAS_PREV "")
  foreach(LIB ${BLAS_LIBRARIES})
    get_filename_component(DIR ${LIB} DIRECTORY)
    if(NOT BLAS_PREV)
      cmake_path(GET DIR PARENT_PATH BLAS_PREV)
    endif()
  endforeach()
  set(BLAS_INCLUDE_SEARCH_PATHS ${BLAS_PREV}/include ${CMAKE_INSTALL_PREFIX}/include ${CMAKE_PREFIX_PATH}/include)
  # Look for mkl in the name of the blas libraries, but not on the path, only the library names
  get_target_property(BLAS_LIBRARIES BLAS::BLAS INTERFACE_LINK_LIBRARIES)
  if(BLAS_LIBRARIES MATCHES "libmkl")
    # Look for mkl.h in includes.
    if(NOT DEFINED MKL_INCLUDE_DIR)
      find_path(MKL_INCLUDE_DIR NAMES mkl.h PATHS ${BLAS_INCLUDE_SEARCH_PATHS})
    endif()
  endif()
  if(DEFINED MKL_INCLUDE_DIR)
    target_compile_definitions(${target_name} PRIVATE USE_MKL)
    target_include_directories(${target_name} PRIVATE ${MKL_INCLUDE_DIR})
  else()
    # Not MKL mode
    target_link_libraries(${target_name} PRIVATE lapacke)
    if (NOT DEFINED BLAS_INCLUDE_DIRS)
      find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h PATHS ${BLAS_INCLUDE_SEARCH_PATHS})
      find_path(BLAS_INCLUDE_DIR NAMES cblas.h PATHS ${BLAS_INCLUDE_SEARCH_PATHS})
    endif()
    target_include_directories(${target_name} PRIVATE ${BLAS_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIR})
  endif()
  target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARY} cufft cublas cusolver curand)
  if (NOT DEFINED UAMMD_INCLUDE_DIRS)
    if (DEFINED uammd_SOURCE_DIR)
      set(UAMMD_INCLUDE_DIRS "${uammd_SOURCE_DIR}/src" "${uammd_SOURCE_DIR}/src/third_party")
    else()
      find_path(UAMMD_INCLUDE_DIR
	NAMES uammd.h
	PATHS ${CMAKE_INSTALL_PREFIX}/include ${CMAKE_PREFIX_PATH}/include
	NO_DEFAULT_PATH
      )
      if (UAMMD_INCLUDE_DIR)
	set(UAMMD_INCLUDE_DIRS "${UAMMD_INCLUDE_DIR}")
      else()
	message(FATAL_ERROR "Could not find UAMMD include directories! \
            Make sure UAMMD is installed or fetched correctly.")
      endif()
    endif()
  endif()
  target_include_directories(${target_name} PRIVATE ${UAMMD_INCLUDE_DIRS})
endfunction()
