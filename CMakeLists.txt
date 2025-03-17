cmake_minimum_required(VERSION 3.10)
project(uammd)
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(UAMMD_MASTER_PROJECT ON)
else()
  set(UAMMD_MASTER_PROJECT OFF)
endif()
if(UAMMD_MASTER_PROJECT)
  file(GLOB_RECURSE headers src/*)

  foreach(header ${headers})
    get_filename_component(header_path ${header} PATH)
    file(RELATIVE_PATH header_path_rel ${CMAKE_CURRENT_SOURCE_DIR}/src ${header_path})
    install(FILES ${header} DESTINATION include/uammd/${header_path_rel})
  endforeach()

  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindUAMMD.cmake"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/UAMMD)

  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/UAMMDSetup.cmake"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/UAMMD
  )
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/UAMMDSetup.cmake)
