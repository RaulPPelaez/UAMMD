include(GoogleTest)
add_executable(tppoisson test_poisson.cu)
add_executable(tppoisson_quadrupole test_tp_quadrupole.cu)
uammd_setup_target(tppoisson)
target_include_directories(tppoisson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tppoisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

uammd_setup_target(tppoisson_quadrupole)
target_include_directories(tppoisson_quadrupole PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tppoisson_quadrupole PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(
  tppoisson
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)

target_link_libraries(
  tppoisson_quadrupole
  PRIVATE
  poisson_common
  GTest::gtest_main
  GTest::gmock_main
)

gtest_discover_tests(tppoisson)
gtest_discover_tests(tppoisson_quadrupole)
