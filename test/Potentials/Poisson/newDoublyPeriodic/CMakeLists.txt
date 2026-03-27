include(GoogleTest)
add_executable(tpDPpoisson test_doublyperiodic_poisson.cu)
uammd_setup_target(tpDPpoisson)
target_include_directories(tpDPpoisson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tpDPpoisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(
    tpDPpoisson
    PRIVATE
    GTest::gtest_main
    GTest::gmock_main
)
gtest_discover_tests(tpDPpoisson)
