include(GoogleTest)
add_executable(poisson test_poisson.cu)
uammd_setup_target(poisson)
target_include_directories(poisson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(poisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(
  poisson
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
gtest_discover_tests(poisson)

add_executable(opoisson Poisson.cu)
uammd_setup_target(opoisson)
set_target_properties(opoisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(rdftest)
