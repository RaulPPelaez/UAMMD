add_executable(testDPDIntegrator testDPDIntegrator.cu)
target_include_directories(testDPDIntegrator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(testDPDIntegrator cufft cublas)
set_target_properties(testDPDIntegrator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  testDPDIntegrator
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(testDPDIntegrator)
