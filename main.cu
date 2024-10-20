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
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<math.h>

#define BLOCK_THREAD_X 8
#define BLOCK_THREAD_Y 8

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " left_image_path right_image_path disparity_image_path" << std::endl;
        return 1;
    }
    std::string l_fileName = argv[1];
    std::string r_fileName = argv[2];
    std::string disp_fileName = argv[3];
    std::string iter_str = argv[4];
    int iter_num = std::atoi(iter_str.c_str());

    cv::Mat l_image = cv::imread(l_fileName, 0);
    cv::Mat r_image = cv::imread(r_fileName, 0);
    
    cv::Mat disp = cv::imread(disp_fileName, 0);

    int subset = 15;
    int sideW = 5;
    int maxIter = iter_num;
    cv::Mat optDisp = cv::Mat(disp.rows, disp.cols, CV_32FC1);
    CDispOptimizeICGN_GPU disp_optimize;
    disp_optimize.run(l_image, r_image, disp, subset, sideW, maxIter, optDisp);
    
    cv::imwrite("./result.tif", optDisp);

    l_image.release();
    r_image.release();
    disp.release();
    return 0;
}