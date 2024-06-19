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

void modifyImage(cv::Mat &_srcImage, cv::Mat &_resultImage){
    for (int i = 0; i < _srcImage.rows; i++)
    {
        for (int j = 0; j < _srcImage.cols; j++)
        {
            _resultImage.at<cv::Vec3f>(i,j)[0] = *_srcImage.ptr<uchar>(i,j);
            _resultImage.at<cv::Vec3f>(i,j)[1] = i;
            _resultImage.at<cv::Vec3f>(i,j)[2] = j;
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
    cv::Mat l_image_result = cv::Mat(l_image.rows, l_image.cols, CV_32FC3);
    cv::Mat r_imagre_result = cv::Mat(l_image.rows, l_image.cols, CV_32FC3);
    modifyImage(l_image, l_image_result);
    modifyImage(r_image, r_imagre_result);
    cv::imwrite("l_image_result.jpg", l_image_result);
    cv::imwrite("r_imagre_result.jpg", r_imagre_result);
    
    cv::Mat disp = cv::imread(disp_fileName, 0);

    /*int width = 32;
    int height = 32;
    cv::Mat l_image = cv::Mat(height, width, CV_8UC1);
    cv::Mat r_image = cv::Mat(height, width, CV_8UC1);
    calImage(width, height, l_image);
    calImage(width, height, r_image);*/

    int subset = 15;
    int sideW = 5;
    int maxIter = 15;
    cv::Mat optDisp = cv::Mat(disp.rows, disp.cols, CV_32FC1);
    CDispOptimizeICGN_GPU disp_optimize;
    disp_optimize.run(l_image, r_image, disp, subset, sideW, maxIter, optDisp);
    CDispOptimizeICGN_CPU disp_optimize_cpu;
    disp_optimize_cpu.run_old(l_image, r_image, disp, subset, sideW, maxIter, optDisp);

    cv::imwrite("./result.png", optDisp);

    l_image.release();
    r_image.release();
    disp.release();
    return 0;
}