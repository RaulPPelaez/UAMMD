include(GoogleTest)
add_executable(test_dpd test_dpd.cu)
add_compile_definitions(test_dpd PRIVATE DOUBLE_PRECISION)
uammd_setup_target(test_dpd)
target_include_directories(test_dpd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(test_dpd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_target_properties(test_dpd PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20)
set_target_properties(test_dpd PROPERTIES CXX_STANDARD_REQUIRED YES CUDA_STANDARD_REQUIRED YES)
find_package(OpenMP REQUIRED)
target_link_libraries(test_dpd PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(test_dpd PRIVATE tbb)
include(FetchContent)
FetchContent_Declare(
  msd
  GIT_REPOSITORY https://github.com/RaulPPelaez/MeanSquareDisplacement/
  GIT_TAG        3.0.0
  CMAKE_ARGS
  -DBUILD_PYTHON=OFF
  -DBUILD_TESTS=OFF
  -DBUILD_EXECUTABLE=OFF
)
FetchContent_MakeAvailable(msd)


target_link_libraries(
  test_dpd
  PRIVATE
  GTest::gtest_main
  msd::msd
)
gtest_discover_tests(test_dpd)
