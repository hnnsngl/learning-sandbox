# setup build system
cmake_minimum_required (VERSION 3.1.0)
set (CMAKE_VERBOSE_MAKEFILE OFF)

# set compiler options
set (CMAKE_CXX_STANDARD 11)

# set output directories
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
  
# Boost
find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

# OpenCV > 3.0
find_package( OpenCV 3.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIR} )

# OpenMP
find_package( OpenMP )

set (LIBS ${OpenCV_Libs} ${Boost_LIBS})
add_subdirectory(src)
