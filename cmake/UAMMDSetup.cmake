function(uammd_setup_target target_name)
    # Ensure the target uses C++ and CUDA standards
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        target_compile_options(${target_name} PRIVATE --expt-relaxed-constexpr -extended-lambda)
    endif()

    # Find and link necessary libraries
    find_package(BLAS)
    if(BLAS_FOUND)
        message("MKL environment detected")
        target_compile_definitions(${target_name} PUBLIC USE_MKL)
        target_link_libraries(${target_name} PUBLIC BLAS::BLAS)
        find_path(BLAS_INCLUDE_DIRS mkl.h PATHS $ENV{CONDA_PREFIX}/include /usr/include /usr/local/include $ENV{MKLROOT}/include $ENV{BLAS_HOME}/include)
    else()
        unset(BLA_VENDOR)
        find_package(LAPACK REQUIRED)
        find_package(LAPACKE REQUIRED)
        find_package(BLAS REQUIRED)
        target_link_libraries(${target_name} PUBLIC ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
    endif()

    # Set include paths
    target_include_directories(${target_name} PUBLIC ${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS})
    target_link_libraries(${target_name} PUBLIC ${CUDA_LIBRARY})

    # Include UAMMD sources
    target_include_directories(${target_name} PUBLIC ${uammd_SOURCE_DIR}/src ${uammd_SOURCE_DIR}/src/third_party)
endfunction()
