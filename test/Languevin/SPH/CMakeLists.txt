add_executable(testSPHIntegrator testSPHIntegrator.cu)
target_include_directories(testSPHIntegrator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(testSPHIntegrator cufft cublas)
set_target_properties(testSPHIntegrator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  testSPHIntegrator
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(testSPHIntegrator)
