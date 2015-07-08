cmake_minimum_required(VERSION 2.8.4)
project(pomdp-solve)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_custom_target(pomdp-solve COMMAND make -C ${pomdp-solve_SOURCE_DIR}
                         CLION_EXE_DIR=${PROJECT_BINARY_DIR})