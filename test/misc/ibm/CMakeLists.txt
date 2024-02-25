add_executable(ibm test_ibm.cu)
target_include_directories(ibm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ibm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  ibm
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(ibm)
