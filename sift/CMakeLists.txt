cmake_minimum_required(VERSION 3.7)
project(sift)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(sift ${SOURCE_FILES})
target_link_libraries( sift ${OpenCV_LIBS} )