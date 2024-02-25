add_executable(pse_test pse_test.cu)
target_include_directories(pse_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pse_test cufft cublas curand)
set_target_properties(pse_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  pse_test
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(pse_test)
