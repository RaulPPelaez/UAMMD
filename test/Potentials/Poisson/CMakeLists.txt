add_library(poisson_common INTERFACE)

target_include_directories(poisson_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(TriplyPeriodic)
add_subdirectory(DoublyPeriodic)
