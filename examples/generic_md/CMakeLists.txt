add_executable(generic_md generic_simulation.cu)
uammd_setup_target(generic_md)
target_include_directories(generic_md PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(generic_md PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
