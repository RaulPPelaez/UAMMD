add_executable(fastChebyshevTransform fastChebyshevTransform.cu)
target_include_directories(fastChebyshevTransform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fastChebyshevTransform cufft cublas)
set_target_properties(fastChebyshevTransform PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  fastChebyshevTransform
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(fastChebyshevTransform)
