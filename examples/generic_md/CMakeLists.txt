add_executable(generic_md generic_simulation.cu)
target_include_directories(generic_md PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(generic_md cufft cublas cusolver curand)
set_target_properties(generic_md PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
