cmake_minimum_required(VERSION 3.10)
project(lab1)

set(CMAKE_CXX_STANDARD 11)

add_executable(main_float main.cpp)


add_executable(main_double main.cpp)
target_compile_definitions(main_double PRIVATE DOUBLE)
