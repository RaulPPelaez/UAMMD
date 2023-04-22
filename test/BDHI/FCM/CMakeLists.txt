add_executable(fcm_test fcm_test.cu)
target_include_directories(fcm_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fcm_test cufft cublas)
set_target_properties(fcm_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  fcm_test
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(fcm_test)
