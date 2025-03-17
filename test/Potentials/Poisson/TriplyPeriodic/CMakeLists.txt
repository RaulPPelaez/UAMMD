include(GoogleTest)
add_executable(tppoisson test_poisson.cu)
uammd_setup_target(tppoisson)
target_include_directories(tppoisson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tppoisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(
  tppoisson
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
gtest_discover_tests(tppoisson)
