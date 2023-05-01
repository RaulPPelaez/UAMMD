add_executable(quasi2d_test quasi2d_test.cu)
target_include_directories(quasi2d_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quasi2d_test cufft cublas)
set_target_properties(quasi2d_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  quasi2d_test
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(quasi2d_test)
