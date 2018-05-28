cmake_minimum_required(VERSION 3.7)
project(cvMat)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(cv ${SOURCE_FILES})
target_link_libraries( cv ${OpenCV_LIBS} )