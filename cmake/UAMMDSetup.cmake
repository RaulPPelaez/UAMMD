include_guard(GLOBAL)

function(uammd_setup_target target_name)
  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA is required but not enabled! \
        You must call enable_language(CUDA) in your main CMakeLists.txt BEFORE calling uammd_setup_target().")
  endif()
  if (NOT TARGET BLAS::BLAS)
    message(FATAL_ERROR "BLAS is required but not enabled! \
	You must call find_package(BLAS) in your main CMakeLists.txt BEFORE calling uammd_setup_target().")
  endif()
  # Ensure the target has C++ and CUDA standards
  target_compile_features(${target_name} PUBLIC cxx_std_14)
  target_compile_features(${target_name} PUBLIC cuda_std_14)

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(${target_name} PRIVATE --expt-relaxed-constexpr -extended-lambda)
  endif()

  # Find and link necessary libraries
  if(BLAS_FOUND)
    message("MKL environment detected")
    target_compile_definitions(${target_name} PUBLIC USE_MKL)
    target_link_libraries(${target_name} PUBLIC BLAS::BLAS)
    find_path(BLAS_INCLUDE_DIRS mkl.h PATHS $ENV{CONDA_PREFIX}/include /usr/include /usr/local/include $ENV{MKLROOT}/include $ENV{BLAS_HOME}/include)
  else()
    unset(BLA_VENDOR)
    find_package(LAPACK REQUIRED)
    find_package(LAPACKE REQUIRED)
    target_link_libraries(${target_name} PUBLIC ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
  endif()

  # Set include paths
  target_include_directories(${target_name} PUBLIC ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS})
  target_link_libraries(${target_name} PUBLIC ${CUDA_LIBRARY})
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
  target_include_directories(${target_name} PUBLIC ${UAMMD_INCLUDE_DIRS})
endfunction()
