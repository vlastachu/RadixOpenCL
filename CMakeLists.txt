cmake_minimum_required(VERSION 3.5)
project(myopencl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp main.h cl.h)

#file(COPY "kernel.c" DESTINATION ${CMAKE_BINARY_DIR})

add_executable(myopencl ${SOURCE_FILES})
target_link_libraries(myopencl OpenCL)