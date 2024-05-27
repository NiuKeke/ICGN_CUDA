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
int main(int argc, char **argv){

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
    cv::Mat float_disp = cv::Mat(disp.rows, disp.cols, CV_32FC1);
    disp.copyTo(float_disp);
    int subset = 15;
    int sideW = 5;
    int maxIter = 15;
    cv::Mat optDisp = cv::Mat(disp.rows, disp.cols, CV_32FC1);
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