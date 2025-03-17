add_executable(test_bvp test_bvp.cu)
uammd_setup_target(test_bvp)
set_target_properties(test_bvp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  test_bvp
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
gtest_discover_tests(test_bvp)
