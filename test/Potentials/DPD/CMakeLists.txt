include(GoogleTest)
add_executable(test_dpd test_dpd.cu)
uammd_setup_target(test_dpd)
target_include_directories(test_dpd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(test_dpd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(
  test_dpd
  PRIVATE
  GTest::gtest_main
)
gtest_discover_tests(test_dpd)
