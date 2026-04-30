include(GoogleTest)
add_executable(dppoisson_quadrupole test_dp_quadrupole.cu)
add_executable(dppoisson test_poisson.cu)
uammd_setup_target(dppoisson_quadrupole)
target_include_directories(dppoisson_quadrupole PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(dppoisson_quadrupole PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

uammd_setup_target(dppoisson)
target_include_directories(dppoisson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(dppoisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(
    dppoisson_quadrupole
    PRIVATE
    poisson_common
    GTest::gtest_main
    GTest::gmock_main
)

target_link_libraries(
    dppoisson
    PRIVATE
    GTest::gtest_main
    GTest::gmock_main
)

gtest_discover_tests(dppoisson_quadrupole)
gtest_discover_tests(dppoisson)
