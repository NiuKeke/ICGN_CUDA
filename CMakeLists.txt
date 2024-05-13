cmake_minimum_required(VERSION 3.24)
project(ICGN_CUDA)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(ICGN_CUDA
        CDispOptICGN.cu
        CDispOptICGN.cuh)
