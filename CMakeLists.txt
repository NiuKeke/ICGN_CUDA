cmake_minimum_required(VERSION 3.28)
project(disp_opt CUDA)

set(CMAKE_CUDA_STANDARD 17)
#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
        disp_optimize_icgn.cu
        disp_optimize_icgn.cuh
        disp_optimize_icgn_cpu.cpp
        disp_optimize_icgn_cpu.h
        main.cu)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
message("OpenCV_LIBS: " ${OpenCV_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

