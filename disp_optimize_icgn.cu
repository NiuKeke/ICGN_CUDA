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
    _y_gradient_image[index] = result;
    result = 0.0f;
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
    _x_gradient_image[index] = result;
}

__global__ void calHessianMat_kernel(int subset, int sideW, int width, int height, float *_x_grad_image, float *_y_grad_image,
                                     double *_hessian_mat)
{
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X + threadIdx.x;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y + threadIdx.y;
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);

    // if (blockIdx.y == 0 && blockIdx.x == 38)
    // {
    //     //printf("blockIdx.x: %d, blockIdx.y: %d, g_x: %d, g_y: %d\n", blockIdx.x, blockIdx.y, g_x, g_y);
    // }
    __shared__ float _x_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k
    __shared__ float _y_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k
    for (int i = 0; i < NUM_PER_THREAD_Y; i++)
    {
        for (int j = 0; j < NUM_PER_THREAD_X; j++)
        {
            _x_grad_image_sm[(threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X] =
                _x_grad_image[(g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X];

            _y_grad_image_sm[(threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X] =
                _y_grad_image[(g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X];

            // if (blockIdx.y == 0 && blockIdx.x == 38)
            // {
            //     int index_region = (threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X;
            //     int row_region = index_region / 32;
            //     int col_region = index_region % 32;
            //     // if (row_region == 18)
            //     // {
            //     //     printf("threadIdx.x: %d, threadIdx.y: %d, row: %d, col: %d, blockIdx.x: %d, blockIdx.y: %d, g_x: %d, g_y: %d,row_region: %d, col_region: %d, _x_grad_image_sm[%d]:%lf, _x_grad_image[%d]: %lf\n",
            //     //            threadIdx.x, threadIdx.y, i, j, blockIdx.x, blockIdx.y, g_x, g_y, row_region, col_region,
            //     //            (threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X,
            //     //            _x_grad_image_sm[(threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x + j * BLOCK_THREAD_DIM_X],
            //     //            (g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X,
            //     //            _x_grad_image[(g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X]);
            //     // }
            // }
        }
    }
    __syncthreads();
    // for (int j = -halfSubset; j <= halfSubset; j++) // y
    // {
    //     for (int k = -halfSubset; k <= halfSubset; k++) // x
    //     {
    //         if (blockIdx.y == 0 && blockIdx.x == 40)
    //         {
    //             printf("threadIdx.x: %d, threadIdx.y: %d, row: %d, col: %d, blockIdx.x: %d, blockIdx.y: %d, g_x: %d, g_y: %d,_x_grad_image_sm[%d]:%lf, _y_grad_image_sm[%d]: %lf\n",
    //                    threadIdx.x, threadIdx.y, j, k, blockIdx.x, blockIdx.y, g_x, g_y,
    //                    halfWinSize * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + threadIdx.y * BLOCK_THREAD_DIM_X + j + halfWinSize + k + threadIdx.x,
    //                    _x_grad_image_sm[halfWinSize * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + threadIdx.y * BLOCK_THREAD_DIM_X + j + halfWinSize + k + threadIdx.x],
    //                    halfWinSize * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + threadIdx.y * BLOCK_THREAD_DIM_X + j + halfWinSize + k + threadIdx.x,
    //                    _y_grad_image_sm[halfWinSize * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + threadIdx.y * BLOCK_THREAD_DIM_X + j + halfWinSize + k + threadIdx.x]);
    //         }
    //     }
    // }

    double hessian[6 * 6] = {0};
    // if ((g_x - halfWinSize) >= 0 && (g_x + halfWinSize) < width && (g_y - halfWinSize) >= 0 && (g_y + halfWinSize) < height)
    {
        for (int j = -halfSubset; j <= halfSubset; j++) // y
        {
            for (int k = -halfSubset; k <= halfSubset; k++) // x
            {
                double Jacobian[6];
                Jacobian[0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + j + threadIdx.x];
                Jacobian[1] = Jacobian[0] * double(k) / double(halfSubset + 1);//x;
                Jacobian[2] = Jacobian[0] * double(j) / double(halfSubset + 1);//y;
                Jacobian[3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + halfWinSize + j + threadIdx.x];
                Jacobian[4] = Jacobian[3] * double(k) / double(halfSubset + 1);
                Jacobian[5] = Jacobian[3] * double(j) / double(halfSubset + 1);

                // if (blockIdx.y == 0 && blockIdx.x == 38)
                // {
                //     int index_region = (halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + j + threadIdx.x;
                //     int row_region = index_region / 32;
                //     int col_region = index_region % 32;
                //     if (row_region == 18 && threadIdx.x == 5 && threadIdx.y == 1)
                //     {
                //         // printf("111 threadIdx.x: %d, threadIdx.y: %d, row: %d, col: %d, blockIdx.x: %d, blockIdx.y: %d, g_x: %d, g_y: %d,row_region: %d, col_region: %d, _x_grad_image_sm[%d]:%lf, _y_grad_image_sm[%d]: %lf\n",
                //         //        threadIdx.x, threadIdx.y, k, j, blockIdx.x, blockIdx.y, g_x, g_y, row_region, col_region,
                //         //        index_region,
                //         //        _x_grad_image_sm[index_region],
                //         //        index_region,
                //         //        _y_grad_image_sm[index_region]);
                //         // printf("Jacobian[0]: %lf, Jacobian[1]: %lf,Jacobian[2]: %lf,Jacobian[3]: %lf,Jacobian[4]: %lf,Jacobian[5]: %lf, localSub_x:%lf,localSub_y: %lf\n",
                //         //  Jacobian[0], Jacobian[1], Jacobian[2],
                //         //  Jacobian[3], Jacobian[4], Jacobian[5], float(k) / float(halfSubset + 1), float(j) / float(halfSubset + 1));
                //     }
                // }

                hessian[0] += Jacobian[0] * Jacobian[0];
                hessian[1] += Jacobian[0] * Jacobian[1];
                hessian[2] += Jacobian[0] * Jacobian[2];
                hessian[3] += Jacobian[0] * Jacobian[3];
                hessian[4] += Jacobian[0] * Jacobian[4];
                hessian[5] += Jacobian[0] * Jacobian[5];
                hessian[6] += Jacobian[1] * Jacobian[0];
                hessian[7] += Jacobian[1] * Jacobian[1];
                hessian[8] += Jacobian[1] * Jacobian[2];
                hessian[9] += Jacobian[1] * Jacobian[3];
                hessian[10] += Jacobian[1] * Jacobian[4];
                hessian[11] += Jacobian[1] * Jacobian[5];
                hessian[12] += Jacobian[2] * Jacobian[0];
                hessian[13] += Jacobian[2] * Jacobian[1];
                hessian[14] += Jacobian[2] * Jacobian[2];
                hessian[15] += Jacobian[2] * Jacobian[3];
                hessian[16] += Jacobian[2] * Jacobian[4];
                hessian[17] += Jacobian[2] * Jacobian[5];
                hessian[18] += Jacobian[3] * Jacobian[0];
                hessian[19] += Jacobian[3] * Jacobian[1];
                hessian[20] += Jacobian[3] * Jacobian[2];
                hessian[21] += Jacobian[3] * Jacobian[3];
                hessian[22] += Jacobian[3] * Jacobian[4];
                hessian[23] += Jacobian[3] * Jacobian[5];
                hessian[24] += Jacobian[4] * Jacobian[0];
                hessian[25] += Jacobian[4] * Jacobian[1];
                hessian[26] += Jacobian[4] * Jacobian[2];
                hessian[27] += Jacobian[4] * Jacobian[3];
                hessian[28] += Jacobian[4] * Jacobian[4];
                hessian[29] += Jacobian[4] * Jacobian[5];
                hessian[30] += Jacobian[5] * Jacobian[0];
                hessian[31] += Jacobian[5] * Jacobian[1];
                hessian[32] += Jacobian[5] * Jacobian[2];
                hessian[33] += Jacobian[5] * Jacobian[3];
                hessian[34] += Jacobian[5] * Jacobian[4];
                hessian[35] += Jacobian[5] * Jacobian[5];
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < 6 * 6; i++)
    {
        _hessian_mat[(g_y + halfWinSize) * width + g_x + halfWinSize + i * width * height] = hessian[i];

        // if (blockIdx.y == 0 && blockIdx.x == 38 && threadIdx.x == 5 && threadIdx.y == 1)
        // {
        //     int hessian_index = (g_y + halfWinSize) * width + g_x + halfWinSize + i * width * height;
        //     printf("blockIdx.x: %d, blockIdx.y: %d, g_x: %d, g_y: %d, hessian_index: %d,hessian[%d]: %lf\n", blockIdx.x, blockIdx.y, g_x, g_y,
        //            hessian_index,i, hessian[i]);
        // }
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

    double *hessian = nullptr;
    generate_hessian_mat(subset, sideW, maxIter, _l_image.cols, _l_image.rows, _x_gradient_image, _y_gradient_image, hessian);

    cv::Mat hessianMat = cv::Mat(36, _l_image.rows * _l_image.cols, CV_64FC1);
    cudaMemcpy(hessianMat.data, hessian, _l_image.rows * _l_image.cols * sizeof(double) * 36,
               cudaMemcpyDeviceToHost);
    for (int n = 0; n < 36; n++)
    {

        double value = hessianMat.data[n * _l_image.rows * _l_image.cols + 16961];
        //printf("n: %d, value: %lf\n", n, value);
    }
    cv::imwrite("./hessian.tif", hessianMat);
}

void CDispOptimizeICGN_GPU::generate_hessian_mat(int subset, int sideW, int maxIter, int width, int height, float *_x_gradient_image,
                                                 float *_y_gradient_image, double *&_hessian_mat)
{
    cudaMalloc((void **)&_hessian_mat, width * height * sizeof(double) * 6 * 6);

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;

    dim3 threads(8, 8);
    dim3 blocks((width - 2 * halfWinSize + threads.x - 1) / (threads.x),
                (height - 2 * halfWinSize + threads.y - 1) / (threads.y));

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