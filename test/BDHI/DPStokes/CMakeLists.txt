add_executable(dpstokes_test dpstokes_test.cu)
uammd_setup_target(dpstokes_test)
target_compile_features(dpstokes_test PRIVATE cxx_std_20)
target_compile_features(dpstokes_test PRIVATE cuda_std_20)
target_include_directories(dpstokes_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(dpstokes_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  dpstokes_test
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(dpstokes_test)
