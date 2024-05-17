cmake_minimum_required(VERSION 3.16)
project(disp_opt CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
enable_language(CUDA)
#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
file(GLOB ${PROJECT_NAME}_SRC_LST 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
add_executable(disp_opt ${${PROJECT_NAME}_SRC_LST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
message("OpenCV_LIBS: " ${OpenCV_LIBS})
set_target_properties(disp_opt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
