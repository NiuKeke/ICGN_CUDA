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
using namespace std;
#define N 3 // 测试矩阵维数定义
//矩阵乘法
void mul(double A[N][N],double B[N][N], double C[N][N])
{
     for(int i=0;i<N;i++)
     {
         for(int j=0;j<N;j++)
         {
             for(int k=0;k<N;k++)
             {
                 C[i][j] += A[i][k]*B[k][j];
             }
         }
     }
 
     //若绝对值小于10^-10,则置为0（这是我自己定的）
     for (int i = 0; i < N * N; i++)
     {
         for (int j = 0; j < N; j++)
         {
             if (abs(C[i][j]) < pow(10, -10))
             {
                 C[i][j] = 0;
             }
         }
     }
}

void inverse3x3(double a[3][3], double inv[3][3]) {
    double det = a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) -
                 a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2]) +
                 a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]);
 
    inv[0][0] = (a[1][1] * a[2][2] - a[2][1] * a[1][2]) / det;
    inv[0][1] = (a[2][1] * a[0][2] - a[0][1] * a[2][2]) / det;
    inv[0][2] = (a[0][1] * a[1][2] - a[1][1] * a[0][2]) / det;
    inv[1][0] = (a[2][0] * a[1][2] - a[1][0] * a[2][2]) / det;
    inv[1][1] = (a[0][0] * a[2][2] - a[2][0] * a[0][2]) / det;
    inv[1][2] = (a[1][0] * a[0][2] - a[0][0] * a[1][2]) / det;
    inv[2][0] = (a[1][0] * a[2][1] - a[2][0] * a[1][1]) / det;
    inv[2][1] = (a[2][0] * a[0][1] - a[0][0] * a[2][1]) / det;
    inv[2][2] = (a[0][0] * a[1][1] - a[1][0] * a[0][1]) / det;
}

void inverse3x3(double a[6][6], double inv[6][6]) {
    double det = a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) -
                 a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2]) +
                 a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]);
 
    inv[0][0] = (a[1][1] * a[2][2] - a[2][1] * a[1][2]) / det;
    inv[0][1] = (a[2][1] * a[0][2] - a[0][1] * a[2][2]) / det;
    inv[0][2] = (a[0][1] * a[1][2] - a[1][1] * a[0][2]) / det;
    inv[1][0] = (a[2][0] * a[1][2] - a[1][0] * a[2][2]) / det;
    inv[1][1] = (a[0][0] * a[2][2] - a[2][0] * a[0][2]) / det;
    inv[1][2] = (a[1][0] * a[0][2] - a[0][0] * a[1][2]) / det;
    inv[2][0] = (a[1][0] * a[2][1] - a[2][0] * a[1][1]) / det;
    inv[2][1] = (a[2][0] * a[0][1] - a[0][0] * a[2][1]) / det;
    inv[2][2] = (a[0][0] * a[1][1] - a[1][0] * a[0][1]) / det;
}
int main(int argc, char **argv)
{
    // double A[N][N] = {0};

    // srand((unsigned)time(0));
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         A[i][j] = rand() % 100 * 0.01;
    //     }
    // }

    // double E_test[N][N] ;
    // double invOfA[N][N] ;//= new double[N * N]();
    // inverse3x3(A, invOfA);

    // mul(A, invOfA, E_test); // 验证精确度

    // cout << "矩阵A:" << endl;
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         cout << A[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << "inv_A:" << endl;
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         cout << invOfA[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << "E_test:" << endl;
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         cout << E_test[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // return 0;
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