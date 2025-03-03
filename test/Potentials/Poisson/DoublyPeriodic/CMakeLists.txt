add_executable(dppoisson PoissonSlab.cu)
uammd_setup_target(dppoisson)
target_include_directories(dppoisson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(dppoisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
