cmake_minimum_required(VERSION 3.10)
project(uammd)
# Install headers
file(GLOB_RECURSE headers src/*)

foreach(header ${headers})
	get_filename_component(header_path ${header} PATH)
	file(RELATIVE_PATH header_path_rel ${CMAKE_CURRENT_SOURCE_DIR}/src ${header_path})
	install(FILES ${header} DESTINATION include/uammd/${header_path_rel})
endforeach()

# Install FindUAMMD.cmake
install(FILES cmake/FindUAMMD.cmake DESTINATION share/cmake/Modules)

message("UAMMD is now being included as a subproject")
if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  message("UAMMD is the main project")
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/UAMMDSetup.cmake)
endif()
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/UAMMDSetup.cmake"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/UAMMD
)
