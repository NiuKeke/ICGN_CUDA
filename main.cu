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

void calImage(int width, int height, cv::Mat &_mat) {
    for (int iRow = 0; iRow < height; iRow++) {
        for (int iCol = 0; iCol < width; iCol++) {
            *_mat.ptr<uchar>(iRow, iCol) = iRow * width + iCol;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " left_image_path right_image_path disparity_image_path" << std::endl;
        return 1;
    }
    std::string l_fileName = argv[1];
    std::string r_fileName = argv[2];
    std::string disp_fileName = argv[3];

    cv::Mat l_image = cv::imread(l_fileName, 0);
    cv::Mat r_image = cv::imread(r_fileName, 0);
    cv::Mat disp = cv::imread(disp_fileName, 0);
    cv::Mat float_disp = cv::Mat(disp.rows, disp.cols, CV_32FC1);
    disp.copyTo(float_disp);

    /*int width = 32;
    int height = 32;
    cv::Mat l_image = cv::Mat(height, width, CV_8UC1);
    cv::Mat r_image = cv::Mat(height, width, CV_8UC1);
    calImage(width, height, l_image);
    calImage(width, height, r_image);*/

    cv::waitKey(0);
    int subset = 15;
    int sideW = 5;
    int maxIter = 15;
    cv::Mat optDisp = float_disp.clone();
    CDispOptimizeICGN_GPU disp_optimize;
    disp_optimize.run(l_image, r_image, float_disp, subset, sideW, maxIter, optDisp);
    // CDispOptimizeICGN_CPU disp_optimize_cpu;
    // disp_optimize_cpu.run_old(l_image, r_image, disp, subset, sideW, maxIter, optDisp);

    cv::imwrite("./result.png", optDisp);

    l_image.release();
    r_image.release();
    disp.release();
    return 0;
}