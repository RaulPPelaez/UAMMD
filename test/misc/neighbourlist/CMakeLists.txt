add_executable(CellList CellList.cu)
uammd_setup_target(CellList)
target_include_directories(CellList PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(CellList PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_link_libraries(
  CellList
  PRIVATE
  GTest::gtest_main
  GTest::gmock_main
)
include(GoogleTest)
gtest_discover_tests(CellList)
