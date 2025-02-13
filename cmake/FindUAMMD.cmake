# Looks for the uammd include folder in the system and sets the UAMMD_INCLUDE_DIRS variable.
# Usage:
# find_package(UAMMD REQUIRED)
# include_directories(${UAMMD_INCLUDE_DIRS})
# The include folder can be in the following locations:
# - In the CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}/include/uammd
# - In the CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}/include/uammd
# - In the system include folder: /usr/include/uammd
# - In the user's home folder: ~/uammd/include/uammd
# - In the conda environment: $ENV{CONDA_PREFIX}/include/uammd
# - In the root folder of the project (we are now inside the cmake/ folder).
find_path(UAMMD_INCLUDE_DIRS uammd.cuh HINTS
  ${CMAKE_PREFIX_PATH}/include/uammd
  ${CMAKE_INSTALL_PREFIX}/include/uammd
  $ENV{CONDA_PREFIX}/include/uammd
  /usr/include/uammd
  $ENV{HOME}/uammd/include/uammd
  ../src)

# Add also the folder UAMMD_INCLUDE_DIRS/third_party to the include directories.
if(UAMMD_INCLUDE_DIRS)
  set(UAMMD_INCLUDE_DIRS ${UAMMD_INCLUDE_DIRS} ${UAMMD_INCLUDE_DIRS}/third_party)
endif()
include(UAMMDSetup)
