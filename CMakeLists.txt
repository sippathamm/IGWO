cmake_minimum_required(VERSION 3.27)
project(GWO)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(GWO
        example/FindGlobalMinimum.cpp
        include/GWO.h)
