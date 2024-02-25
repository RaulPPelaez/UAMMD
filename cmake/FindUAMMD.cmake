# Looks for the uammd include folder in the system and sets the UAMMD_INCLUDE_DIR variable.
# Usage:
# find_package(UAMMD REQUIRED)
# include_directories(${UAMMD_INCLUDE_DIR})
# The include folder can be in the following locations:
# 1. In the system include folder: /usr/include/uammd
# 2. In the user's home folder: ~/uammd/include/uammd
# 3. In the conda environment: $ENV{CONDA_PREFIX}/include/uammd

# First, look for the include folder in the system.
find_path(UAMMD_INCLUDE_DIR uammd.cuh HINTS /usr/include/uammd)

# If the include folder is not found, look for it in the user's home folder.
if(NOT UAMMD_INCLUDE_DIR)
  find_path(UAMMD_INCLUDE_DIR uammd.cuh HINTS $ENV{HOME}/uammd/include/uammd)
endif()

# If the include folder is not found, look for it in the conda environment.
if(NOT UAMMD_INCLUDE_DIR)
  find_path(UAMMD_INCLUDE_DIR uammd.cuh HINTS $ENV{CONDA_PREFIX}/include/uammd)
endif()

# Add also the folder UAMMD_INCLUDE_DIR/third_party to the include directories.
if(UAMMD_INCLUDE_DIR)
  set(UAMMD_INCLUDE_DIR ${UAMMD_INCLUDE_DIR} ${UAMMD_INCLUDE_DIR}/third_party)
endif()
