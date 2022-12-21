
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.1
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)

add_executable(neighbour_list neighbour_list.cu )
add_executable(electrostatic_forces electrostatic_forces.cu)
target_link_libraries(electrostatic_forces cufft)
pybind11_add_module(python_wrapper python_wrapper.cu)
set_target_properties(python_wrapper PROPERTIES CUDA_ARCHITECTURES OFF)

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu )
foreach( testsourcefile ${SOURCES} )
    # I used a simple string replace, to cut off .cpp.
    string( REPLACE ".cu" "" testname ${testsourcefile} )
    set_target_properties(${testname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach( testsourcefile ${SOURCES} )
