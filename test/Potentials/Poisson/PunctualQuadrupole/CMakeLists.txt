include(GoogleTest)
add_executable(tpquadrupole test_quadrupole.cu)
uammd_setup_target(tpquadrupole)
target_include_directories(tpquadrupole PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tpquadrupole PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(
  tpquadrupole
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
gtest_discover_tests(tpquadrupole)
