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

void calImage(int width, int height, cv::Mat &_mat){
    for (int iRow = 0; iRow < height; iRow++)
    {
        for (int iCol = 0; iCol < width; iCol++)
        {
            *_mat.ptr<float>(iRow, iCol) = 1.0f;//iRow * width + iCol;
        }
    }
}

__global__ void mean_image(int subset, int sideW, int width, int height, float *_src_image, float *_dst_image){
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % 32;
    int thread_y = thread_index / 32;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * 32;
    int g_y = blockIdx.y * 32;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);
    __shared__ float _src_image_sm[32 * 32]; // 4k

    for (int i = 0; i < 16; ++i)
    {
        _src_image_sm[(thread_y + i * 2) * 32 + thread_x] = 
        _src_image[(g_y + thread_y + i * 2) * width + g_x + thread_x];
    }
    __syncthreads();

    const int lane_id = thread_index % 8;
    
    float sum = 0.0f;
    float buffer = 0;
    for (int j = -halfSubset; j <= halfSubset; j++)
    {
        for (int k = -(halfSubset + lane_id); k < (halfSubset + 8 - lane_id); k++)
        {
            if(lane_id == 0){
                buffer = _src_image_sm[(halfSubset + threadIdx.y + j) * 32 +
                                                  halfSubset + k + threadIdx.x];
            }
            buffer = __shfl_sync(0xFFFFFFFFU, buffer, 0, 8);
            if(k >= -7 && k <= 7){
                sum += buffer;
            }
            // if(j == -halfSubset && threadIdx.x < 8 && threadIdx.y == 0 && lane_id == 3 && k == -7){
            //     printf("111 j: %d, k: %d, lane_id: %d, threadIdx.x: %d, buffer: %lf, num: %d\n",j,k, lane_id,threadIdx.x,  buffer, num);
            // }

            // if(j == -halfSubset && threadIdx.x < 8 && threadIdx.y == 0 && lane_id == 1 && k == -5){
            //     printf("222 j: %d, k: %d, lane_id: %d, threadIdx.x: %d, buffer: %lf, num: %d\n",j,k, lane_id,threadIdx.x,  buffer, num);
            // }

            // if(j == -halfSubset && threadIdx.x < 8 && threadIdx.y == 0 && lane_id == 0 && k == -4){
            //     printf("333 j: %d, k: %d, lane_id: %d, threadIdx.x: %d, buffer: %lf, num: %d\n",j,k, lane_id,threadIdx.x,  buffer, num);
            // }
        }
    }
    float mean_value = float(sum) / float(subset * subset);
    _dst_image[(g_y + halfWinSize + threadIdx.y) * width + g_x + halfWinSize + threadIdx.x] = mean_value;
}

void test_mean_image(cv::Mat &_src_image, cv::Mat &_mean_image){
    float *_src_image_gpu = nullptr;
    cudaMalloc((void**)&_src_image_gpu, sizeof(float) * _src_image.cols * _src_image.rows);
    float *_dst_image_gpu = nullptr;
    cudaMalloc((void**)&_dst_image_gpu,sizeof(float) * _mean_image.cols * _mean_image.rows);

    cudaMemcpy(_src_image_gpu, (float *)_src_image.data, sizeof(float) * _src_image.cols * _src_image.rows, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(_dst_image_gpu, (float *)_mean_image.data, sizeof(float) * _mean_image.cols * _mean_image.rows, cudaMemcpyKind::cudaMemcpyHostToDevice);
    dim3 threads(8, 8);
    dim3 blocks((_src_image.cols - 2 * 12 + threads.x - 1) / (threads.x),
                (_src_image.rows - 2 * 12 + threads.y - 1) / (threads.y));
    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    mean_image<<<blocks, threads>>>(15, 5, _src_image.cols, _src_image.rows, _src_image_gpu, _dst_image_gpu);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("generate_mean_image time: %f ms\n", time);

    cudaMemcpy((float*)_mean_image.data, _dst_image_gpu,sizeof(float) * _mean_image.cols * _mean_image.rows, cudaMemcpyKind::cudaMemcpyDeviceToHost);

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
    int width = 1280;
    int height = 1024;
    cv::Mat image = cv::Mat(height, width, CV_32FC1);
    calImage(width,height, image);
    // cv::imwrite("image.tif", image);
    cv::Mat mean_image = cv::Mat(height, width,CV_32FC1);
    mean_image.setTo(0.0f);
    test_mean_image(image, mean_image);
    cv::imwrite("meanImage.tif", mean_image);
    // if(argc < 4){
    //     std::cerr << "Usage: " << argv[0] << " left_image_path right_image_path disparity_image_path" << std::endl;
    //     return 1;
    // }
    // std::string l_fileName = argv[1];
    // std::string r_fileName = argv[2];
    // std::string disp_fileName = argv[3];

    // cv::Mat l_image = cv::imread(l_fileName, 0);
    // cv::Mat r_image = cv::imread(r_fileName, 0);
    // cv::Mat disp = cv::imread(disp_fileName, 0);
    // cv::Mat float_disp = cv::Mat(disp.rows, disp.cols, CV_32FC1);
    // disp.copyTo(float_disp);
    // int subset = 15;
    // int sideW = 5;
    // int maxIter = 15;
    // cv::Mat optDisp = float_disp.clone();
    // CDispOptimizeICGN_GPU disp_optimize;
    // disp_optimize.run(l_image, r_image, float_disp, subset, sideW, maxIter, optDisp);
    // // CDispOptimizeICGN_CPU disp_optimize_cpu;
    // // disp_optimize_cpu.run_old(l_image, r_image, disp, subset, sideW, maxIter, optDisp);

    // cv::imwrite("./result.png", optDisp);

    // l_image.release();
    // r_image.release();
    // disp.release();
    return 0;
}