add_executable(ParticleSorter ParticleSorter.cu)
uammd_setup_target(ParticleSorter)
target_include_directories(ParticleSorter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ParticleSorter PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  ParticleSorter
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(ParticleSorter)
