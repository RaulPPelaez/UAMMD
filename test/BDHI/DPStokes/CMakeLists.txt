add_executable(dpstokes_test dpstokes_test.cu)
target_include_directories(dpstokes_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dpstokes_test cufft cublas)
set_target_properties(dpstokes_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  dpstokes_test
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(dpstokes_test)
