add_executable(lanczos test_lanczos.cu)
uammd_setup_target(lanczos)
target_include_directories(lanczos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(lanczos PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  lanczos
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(lanczos)
