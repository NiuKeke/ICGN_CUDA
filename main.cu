//
// Created by Administrator on 2024/4/28.
//
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "disp_optimize_icgn_cpu.h"
#include "disp_optimize_icgn.cuh"

void printDeviceProperties(const cudaDeviceProp& prop, int deviceIndex) {
    std::cout << "\nDevice " << deviceIndex << " properties:\n";
    std::cout << "  Name:                               " << prop.name << "\n";
    std::cout << "  Compute Capability:                 " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total Global Memory:                " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "  Shared Memory Per Block:             " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "  Registers Per Block:                 " << prop.regsPerBlock << "\n";
    std::cout << "  Warp Size:                          " << prop.warpSize << "\n";
    std::cout << "  Max Threads Per Block:               " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max Thread Dimensions:               ["
              << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
    std::cout << "  Max Grid Dimensions:                 ["
              << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
    std::cout << "  Total Constant Memory:              " << prop.totalConstMem / 1024 << " KB\n";
    std::cout << "  multiProcessorCount:                  " << prop.multiProcessorCount << std::endl;
    std::cout << "  maxBlocksPerMultiProcessor:           " << prop.maxBlocksPerMultiProcessor<< std::endl;
    std::cout << "  regsPerMultiprocessor:                " << prop.regsPerMultiprocessor<<std::endl;
    std::cout << "  maxThreadsPerMultiProcessor:          " << prop.maxThreadsPerMultiProcessor << std::endl;
    // ... 根据需要继续添加其他属性 ...
}


int main(int argc, char **argv){
//    int deviceCount;
//    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
//
//    if (cudaStatus != cudaSuccess) {
//        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(cudaStatus) << std::endl;
//        return 1;
//    }
//
//    if (deviceCount == 0) {
//        std::cout << "No CUDA capable devices found" << std::endl;
//        return 1;
//    }
//
//    std::cout << "Found " << deviceCount << " CUDA devices:" << std::endl;
//
//    for (int i = 0; i < deviceCount; ++i) {
//        cudaDeviceProp prop;
//        cudaStatus = cudaGetDeviceProperties(&prop, i);
//        if (cudaStatus != cudaSuccess) {
//            std::cerr << "cudaGetDeviceProperties for device " << i << " failed: " << cudaGetErrorString(cudaStatus) << std::endl;
//            continue;
//        }
//        printDeviceProperties(prop, i);
//    }
    if(argc < 4){
        std::cerr << "Usage: " << argv[0] << " left_image_path right_image_path disparity_image_path" << std::endl;
        return 1;
    }
    std::string l_fileName = argv[1];
    std::string r_fileName = argv[2];
    std::string disp_fileName = argv[3];

    cv::Mat l_image = cv::imread(l_fileName, 0);
    cv::Mat r_image = cv::imread(r_fileName, 0);
    cv::Mat disp = cv::imread(disp_fileName, 0);

    int subset = 15;
    int sideW = 5;
    int maxIter = 15;
    cv::Mat optDisp = cv::Mat(disp.rows, disp.cols, CV_32FC1);
    CDispOptimizeICGN_GPU disp_optimize;
    disp_optimize.run(l_image, r_image, disp, subset, sideW, maxIter, optDisp);
    // CDispOptimizeICGN_CPU disp_optimize_cpu;
    // disp_optimize_cpu.run_old(l_image, r_image, disp, subset, sideW, maxIter, optDisp);

    cv::imwrite("./result.png", optDisp);

    l_image.release();
    r_image.release();
    disp.release();
    return 0;
}