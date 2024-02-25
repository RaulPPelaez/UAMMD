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
