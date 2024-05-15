#include "disp_optimize_icgn.cuh"
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#define BLOCK_DATA_DIM_X 32
#define BLOCK_DATA_DIM_Y 32
#define BLOCK_THREAD_DIM_X 8
#define BLOCK_THREAD_DIM_Y 8
#define NUM_PER_THREAD_X 4
#define NUM_PER_THREAD_Y 4
__global__ void generate_gradient_image_kernel(int width, int height, uchar *_src_image,
                                               float *_x_gradient_image, float *_y_gradient_image)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    if (x >= width || y >= height)
    {
        return;
    }

    float result = 0.0f;
    if ((x + 2) >= width || (x - 2) < 0)
    {
        result = 0.0f;
    }
    else
    {
        result -= (float)_src_image[y * width + x + 2] * 0.083333333333333f;
        result += (float)_src_image[y * width + x + 1] * 0.666666666666667f;
        result -= (float)_src_image[y * width + x - 1] * 0.666666666666667f;
        result += (float)_src_image[y * width + x - 2] * 0.083333333333333f;
    }
    _x_gradient_image[index] = result;
    if ((y + 2) >= height || (y - 2) < 0)
    {
        result = 0.0f;
    }
    else
    {
        result -= (float)_src_image[(y + 2) * width + x] * 0.083333333333333f;
        result += (float)_src_image[(y + 1) * width + x] * 0.666666666666667f;
        result -= (float)_src_image[(y - 1) * width + x] * 0.666666666666667f;
        result += (float)_src_image[(y - 2) * width + x] * 0.083333333333333f;
    }
    _y_gradient_image[index] = result;
}

__global__ void calHessianMat_kernel(int subset, int sideW, int width, int height, float *_x_grad_image, float *_y_grad_image,
                                     float *_hessian_mat)
{
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X + threadIdx.x;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y + threadIdx.y;
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    __shared__ float _x_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y];
    __shared__ float _y_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y];
    for (int i = 0; i < NUM_PER_THREAD_Y; i++)
    {
        for (int j = 0; j < NUM_PER_THREAD_X; j++)
        {
            _x_grad_image_sm[(threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X] =
                _x_grad_image[(g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X];

            _y_grad_image_sm[(threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X] =
                _y_grad_image[(g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X];
        }
    }

    // for (int i = 0; i < NUM_PER_THREAD_Y; i++)
    // {
    //     for (int j = 0; j < NUM_PER_THREAD_X; j++)
    //     {
    //         if (blockIdx.x == 0 && blockIdx.y == 0 && i == 0 &&j == 0){
    //             printf("threadX:%d, threadY:%d, i: %d,j: %d,_x_grad_image_sm[%d]:%lf, g_x: %d. g_y: %d,g_image[%d]: %lf\n",
    //                threadIdx.x, threadIdx.y, i,j,
    //                (threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X,
    //                _x_grad_image_sm[(threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X],
    //                g_x,g_y,
    //                (g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X,
    //                _x_grad_image[(g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X]);
    //         }      
    //         // if (blockIdx.x == 0 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0)
    //         // {
    //         //     printf("threadX:%d, threadY:%d, i:%d, j:%d, sm_index:%d, g_x:%d, g_y:%d, g_index1:%d \n",
    //         //            threadIdx.x, threadIdx.y, i, j,
    //         //            (threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X,
    //         //            g_x, g_y,
    //         //            (g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X);
    //         // }
    //     }
    // }

    __syncthreads();

    if ((g_x - halfWinSize) >= 0 && (g_x + halfWinSize) < width && (g_y - halfWinSize) >= 0 && (g_y + halfWinSize) < height)
    {
        
    }
}
void CDispOptimizeICGN_GPU::run(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter, cv::Mat &_result)
{
    // 生成左图像梯度影像,分为x,y两个方向;
    cv::Mat _x_gradient_image_cpu, _y_gradient_image_cpu;
    _x_gradient_image_cpu.create(_l_image.size(), CV_32FC1);
    _y_gradient_image_cpu.create(_l_image.size(), CV_32FC1);
    // generate_gradient_image(_l_image, _x_gradient_image, _y_gradient_image);
    // // 保存梯度影像;
    

    float *_x_gradient_image = nullptr;
    float *_y_gradient_image = nullptr;
    generate_gradient_image(_l_image, _x_gradient_image, _y_gradient_image);

    cudaMemcpy(_x_gradient_image_cpu.data, _x_gradient_image, _l_image.rows * _l_image.cols * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(_y_gradient_image_cpu.data, _y_gradient_image, _l_image.rows * _l_image.cols * sizeof(float),
               cudaMemcpyDeviceToHost);
    cv::imwrite("x_gradient_image_cpu.tif", _x_gradient_image_cpu);
    cv::imwrite("y_gradient_image_cpu.tif", _y_gradient_image_cpu);

    float *hessian = nullptr;
    generate_hessian_mat(subset, sideW, maxIter, _l_image.cols, _l_image.rows, _x_gradient_image, _y_gradient_image, hessian);

    cv::Mat hessianMat = cv::Mat(_l_image.rows, _l_image.cols * 36, CV_32FC1);
    cudaMemcpy(hessianMat.data, hessian, _l_image.rows * _l_image.cols * sizeof(float) * 36,
               cudaMemcpyDeviceToHost);
    cv::imwrite("./hessian.tif", hessianMat);
}

void CDispOptimizeICGN_GPU::generate_hessian_mat(int subset, int sideW, int maxIter, int width, int height, float *_x_gradient_image,
                                                 float *_y_gradient_image, float *_hessian_mat)
{
    cudaMalloc((void **)&_hessian_mat, width * height * sizeof(float) * 6 * 6);

    dim3 threads(8, 8);
    dim3 blocks((width + threads.x * NUM_PER_THREAD_X - 1) / (threads.x * NUM_PER_THREAD_X),
                (height + threads.y * NUM_PER_THREAD_Y - 1) / (threads.y * NUM_PER_THREAD_Y));

    printf("width: %d, height: %d, blocks.x: %d, blocks.y: %d, threads.x: %d, threads.y: %d\n",
           width, height, blocks.x, blocks.y, threads.x, threads.y);
    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    calHessianMat_kernel<<<blocks, threads>>>(subset, sideW, width, height, _x_gradient_image, _y_gradient_image, _hessian_mat);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("generate_hessian_mat time: %f ms\n", time);
}

void CDispOptimizeICGN_GPU::generate_gradient_image(cv::Mat &_l_image, float *&_x_gradient_image, float *&_y_gradient_image)
{
    uchar *src_image = nullptr;
    cudaMalloc((void **)&src_image, _l_image.rows * _l_image.cols * sizeof(uchar));
    cudaMemcpy(src_image, _l_image.data, _l_image.rows * _l_image.cols * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((_l_image.cols + threads.x - 1) / threads.x, (_l_image.rows + threads.y - 1) / threads.y);

    cudaMalloc((void **)&_x_gradient_image, _l_image.rows * _l_image.cols * sizeof(float));

    cudaMalloc((void **)&_y_gradient_image, _l_image.rows * _l_image.cols * sizeof(float));

    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    generate_gradient_image_kernel<<<blocks, threads>>>(_l_image.cols, _l_image.rows, src_image, _x_gradient_image, _y_gradient_image);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("generate_gradient_image_x time: %f ms\n", time);

    return;
}

void CDispOptimizeICGN_GPU::generate_gradient_image(cv::Mat &_l_image, cv::Mat &_x_gradient_image, cv::Mat &_y_gradient_image)
{
    uchar *src_image = nullptr;
    cudaMalloc((void **)&src_image, _l_image.rows * _l_image.cols * sizeof(uchar));
    cudaMemcpy(src_image, _l_image.data, _l_image.rows * _l_image.cols * sizeof(uchar),
               cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((_l_image.cols + threads.x - 1) / threads.x, (_l_image.rows + threads.y - 1) / threads.y);
    float *_x_dst_image = nullptr;
    cudaMalloc((void **)&_x_dst_image, _l_image.rows * _l_image.cols * sizeof(float));
    float *_y_dst_image = nullptr;
    cudaMalloc((void **)&_y_dst_image, _l_image.rows * _l_image.cols * sizeof(float));

    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    generate_gradient_image_kernel<<<blocks, threads>>>(_l_image.cols, _l_image.rows, src_image, _x_dst_image, _y_dst_image);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("generate_gradient_image_x time: %f ms\n", time);
    cudaMemcpy(_x_gradient_image.data, _x_dst_image, _l_image.rows * _l_image.cols * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(_y_gradient_image.data, _y_dst_image, _l_image.rows * _l_image.cols * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(src_image);
    cudaFree(_x_dst_image);
    cudaFree(_y_dst_image);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return;
}