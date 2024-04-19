cmake_minimum_required(VERSION 3.27)
project(IGWO)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(IGWO
        include/BenchmarkFunction.h
        include/IGWO.h
        example/GlobalMinimum.cpp
)
