#include "disp_optimize_icgn.cuh"
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#define BLOCK_DATA_DIM_X 32
#define BLOCK_DATA_DIM_Y 32
#define SUB_THREAD_DIM_X 4
#define SUB_THREAD_DIM_Y 4
#define BLOCK_THREAD_DIM_X 8
#define BLOCK_THREAD_DIM_Y 8
#define NUM_PER_THREAD_X 4
#define NUM_PER_THREAD_Y 4
#define WARP_SIZE 32
#define SUBSET_SIZE 15
#define SUBREGION_NUM (SUBSET_SIZE * SUBSET_SIZE)
__constant__ float MBT[4][4];
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

__global__ void
calHessianMat_kernel(int subset, int sideW, int width, int height, float *_x_grad_image, float *_y_grad_image,
                     double *_hessian_mat)
{
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X + threadIdx.x;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y + threadIdx.y;
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);

    __shared__ float _x_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k
    __shared__ float _y_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k
    for (int i = 0; i < NUM_PER_THREAD_Y; i++)
    {
        for (int j = 0; j < NUM_PER_THREAD_X; j++)
        {
            _x_grad_image_sm[(threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x +
                             j * BLOCK_THREAD_DIM_X] =
                _x_grad_image[(g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X];

            _y_grad_image_sm[(threadIdx.y + i * BLOCK_THREAD_DIM_Y) * blockDim.x * NUM_PER_THREAD_X + threadIdx.x +
                             j * BLOCK_THREAD_DIM_X] =
                _y_grad_image[(g_y + i * BLOCK_THREAD_DIM_Y) * width + g_x + j * BLOCK_THREAD_DIM_X];
        }
    }
    __syncthreads();

    double hessian[6 * 6] = {0};
    // if ((g_x - halfWinSize) >= 0 && (g_x + halfWinSize) < width && (g_y - halfWinSize) >= 0 && (g_y + halfWinSize) < height)
    {
        for (int j = -halfSubset; j <= halfSubset; j++) // y
        {
            for (int k = -halfSubset; k <= halfSubset; k++) // x
            {
                double Jacobian[6];
                Jacobian[0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + j) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                               halfWinSize + k + threadIdx.x];
                Jacobian[1] = Jacobian[0] * double(j) / double(halfSubset + 1); // x;
                Jacobian[2] = Jacobian[0] * double(k) / double(halfSubset + 1); // y;
                Jacobian[3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + j) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X +
                                               halfWinSize + k + threadIdx.x];
                Jacobian[4] = Jacobian[3] * double(j) / double(halfSubset + 1);
                Jacobian[5] = Jacobian[3] * double(k) / double(halfSubset + 1);

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
        //_hessian_mat[i] = hessian[i];
    }
}

__global__ void
calHessianMat_kernel_opt(int subset, int sideW, int width, int height, float *_x_grad_image, float *_y_grad_image,
                         double *_hessian_mat)
{

    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % BLOCK_DATA_DIM_X;
    int thread_y = thread_index / BLOCK_DATA_DIM_X;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);

    __shared__ float _x_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k
    __shared__ float _y_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k

    for (int i = 0; i < WARP_SIZE / 2; ++i)
    {
        _x_grad_image_sm[(thread_y * 16 + i) * BLOCK_DATA_DIM_X + thread_x] = _x_grad_image[(g_y + thread_y * 16 + i) * width + g_x + thread_x];
        _y_grad_image_sm[(thread_y * 16 + i) * BLOCK_DATA_DIM_X + thread_x] = _y_grad_image[(g_y + thread_y * 16 + i) * width + g_x + thread_x];
    }

    __syncthreads();

    double hessian[6 * 6] = {0};
    // if ((g_x - halfWinSize) >= 0 && (g_x + halfWinSize) < width && (g_y - halfWinSize) >= 0 && (g_y + halfWinSize) < height)
    {
        for (int j = -halfSubset; j <= halfSubset; j++) // y
        {
            for (int k = -halfSubset; k <= halfSubset; k++) // x
            {
                double Jacobian[6];
                Jacobian[0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                               halfWinSize + j + threadIdx.x];
                Jacobian[1] = Jacobian[0] * double(k) / double(halfSubset + 1); // x;
                Jacobian[2] = Jacobian[0] * double(j) / double(halfSubset + 1); // y;
                Jacobian[3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X +
                                               halfWinSize + j + threadIdx.x];
                Jacobian[4] = Jacobian[3] * double(k) / double(halfSubset + 1);
                Jacobian[5] = Jacobian[3] * double(j) / double(halfSubset + 1);

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
        _hessian_mat[(g_y + halfWinSize + threadIdx.y) * width + g_x + threadIdx.x + halfWinSize +
                     i * width * height] = hessian[i];
        //_hessian_mat[i] = hessian[i];
    }
}

__global__ void calHessianMat_kernel_opt_write_back(int subset, int sideW, int width, int height, float *_x_grad_image,
                                                    float *_y_grad_image,
                                                    double *_hessian_mat)
{

    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % BLOCK_DATA_DIM_X;
    int thread_y = thread_index / BLOCK_DATA_DIM_X;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);

    __shared__ float mem_sm[BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y * 6 * 6];
    float *_x_grad_image_sm = &mem_sm[0];
    float *_y_grad_image_sm = &mem_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y];
    float *_hessian_mat_sm = &mem_sm[0];
    for (int i = 0; i < WARP_SIZE / 2; ++i)
    {
        _x_grad_image_sm[(thread_y + i * 2) * BLOCK_DATA_DIM_X + thread_x] = _x_grad_image[(g_y + thread_y + i * 2) * width + g_x + thread_x];
        _y_grad_image_sm[(thread_y + i * 2) * BLOCK_DATA_DIM_X + thread_x] = _y_grad_image[(g_y + thread_y + i * 2) * width + g_x + thread_x];
    }

    __syncthreads();

    double hessian[6 * 6] = {0};
    // if ((g_x - halfWinSize) >= 0 && (g_x + halfWinSize) < width && (g_y - halfWinSize) >= 0 && (g_y + halfWinSize) < height)
    {
        for (int j = -halfSubset; j <= halfSubset; j++) // y
        {
            for (int k = -halfSubset; k <= halfSubset; k++) // x
            {
                double Jacobian[6];
                Jacobian[0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + j) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                               halfWinSize + k + threadIdx.x];
                Jacobian[1] = Jacobian[0] * double(j) / double(halfSubset + 1); // x;
                Jacobian[2] = Jacobian[0] * double(k) / double(halfSubset + 1); // y;
                Jacobian[3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + j) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X +
                                               halfWinSize + k + threadIdx.x];
                Jacobian[4] = Jacobian[3] * double(j) / double(halfSubset + 1);
                Jacobian[5] = Jacobian[3] * double(k) / double(halfSubset + 1);

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
    int block_index = blockIdx.y * gridDim.x + blockIdx.x;

    for (int i = 0; i < 6 * 6; i++)
    {
        int g_index = (halfWinSize)*width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y +
                      thread_index + i * width * height;
        _hessian_mat[g_index] = hessian[i];
    }
}

__global__ void
calHessianMat_kernel_opt_write_back_opt(int subset, int sideW, int width, int height, float *_x_grad_image,
                                        float *_y_grad_image,
                                        float *_hessian_mat)
{

    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % BLOCK_DATA_DIM_X;
    int thread_y = thread_index / BLOCK_DATA_DIM_X;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);

    __shared__ float mem_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y * 2];
    float *_x_grad_image_sm = &mem_sm[0];
    float *_y_grad_image_sm = &mem_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y];

    for (int i = 0; i < WARP_SIZE / 2; ++i)
    {
        _x_grad_image_sm[(thread_y + i * 2) * BLOCK_DATA_DIM_X + thread_x] = _x_grad_image[(g_y + thread_y + i * 2) * width + g_x + thread_x];
        _y_grad_image_sm[(thread_y + i * 2) * BLOCK_DATA_DIM_X + thread_x] = _y_grad_image[(g_y + thread_y + i * 2) * width + g_x + thread_x];
    }

    __syncthreads();
    const int lane_id = thread_index % 8;
    float hessian[6 * 6] = {0};

    // if ((g_x - halfWinSize) >= 0 && (g_x + halfWinSize) < width && (g_y - halfWinSize) >= 0 && (g_y + halfWinSize) < height)
    {
        for (int j = -halfSubset; j <= halfSubset; j++) // y
        {
            for (int k = -(halfSubset + lane_id); k < (halfSubset + 8 - lane_id); k++) // x
            {
                float Jacobian[6] = {0};
                float coord_weight[2] = {0};
                if (lane_id == 0)
                {
                    int index = (halfWinSize + threadIdx.y + j) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + k +
                                threadIdx.x;
                    Jacobian[0] = _x_grad_image_sm[index];
                    Jacobian[3] = _y_grad_image_sm[index];
                }
                Jacobian[0] = __shfl_sync(0xFFFFFFFFU, Jacobian[0], 0, 8);
                Jacobian[3] = __shfl_sync(0xFFFFFFFFU, Jacobian[3], 0, 8);

                coord_weight[0] = float(j) / float(halfSubset + 1);
                coord_weight[1] = float(k) / float(halfSubset + 1);
                if (k >= -7 && k <= 7)
                {

                    Jacobian[1] = Jacobian[0] * coord_weight[0]; // x;
                    Jacobian[2] = Jacobian[0] * coord_weight[1]; // y;
                    Jacobian[4] = Jacobian[3] * coord_weight[0];
                    Jacobian[5] = Jacobian[3] * coord_weight[1];

                    float tmp[6] = {0};
                    tmp[0] = Jacobian[0] * Jacobian[0];
                    tmp[1] = Jacobian[0] * Jacobian[1];
                    tmp[2] = Jacobian[0] * Jacobian[2];
                    tmp[3] = Jacobian[0] * Jacobian[3];
                    tmp[4] = Jacobian[0] * Jacobian[4];
                    tmp[5] = Jacobian[0] * Jacobian[5];

                    hessian[0] += tmp[0];
                    hessian[1] += tmp[1];
                    hessian[2] += tmp[2];
                    hessian[3] += tmp[3];
                    hessian[4] += tmp[4];
                    hessian[5] += tmp[5];

                    hessian[6] += tmp[1];
                    hessian[12] += tmp[2];
                    hessian[18] += tmp[3];
                    hessian[24] += tmp[4];
                    hessian[30] += tmp[5];

                    tmp[1] = Jacobian[1] * Jacobian[1];
                    tmp[2] = Jacobian[1] * Jacobian[2];
                    tmp[3] = Jacobian[1] * Jacobian[3];
                    tmp[4] = Jacobian[1] * Jacobian[4];
                    tmp[5] = Jacobian[1] * Jacobian[5];

                    hessian[7] += tmp[1];
                    hessian[8] += tmp[2];
                    hessian[9] += tmp[3];
                    hessian[10] += tmp[4];
                    hessian[11] += tmp[5];

                    hessian[13] += tmp[2];
                    hessian[19] += tmp[3];
                    hessian[25] += tmp[4];
                    hessian[31] += tmp[5];

                    tmp[2] = Jacobian[2] * Jacobian[2];
                    tmp[3] = Jacobian[2] * Jacobian[3];
                    tmp[4] = Jacobian[2] * Jacobian[4];
                    tmp[5] = Jacobian[2] * Jacobian[5];

                    hessian[14] += tmp[2];
                    hessian[15] += tmp[3];
                    hessian[16] += tmp[4];
                    hessian[17] += tmp[5];

                    hessian[20] += tmp[3];
                    hessian[26] += tmp[4];
                    hessian[32] += tmp[5];

                    tmp[3] = Jacobian[3] * Jacobian[3];
                    tmp[4] = Jacobian[3] * Jacobian[4];
                    tmp[5] = Jacobian[3] * Jacobian[5];

                    hessian[21] += tmp[3];
                    hessian[22] += tmp[4];
                    hessian[23] += tmp[5];

                    hessian[27] += tmp[4];
                    hessian[33] += tmp[5];

                    tmp[4] = Jacobian[4] * Jacobian[4];
                    tmp[5] = Jacobian[4] * Jacobian[5];

                    hessian[28] += tmp[4];
                    hessian[29] += tmp[5];

                    hessian[34] += tmp[5];

                    tmp[5] = Jacobian[5] * Jacobian[5];
                    hessian[35] += tmp[5];
                }
            }
        }
    }

    __syncthreads();
    int block_index = blockIdx.y * gridDim.x + blockIdx.x;

    for (int i = 0; i < 6 * 6; i++)
    {
        int g_index = (halfWinSize)*width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y +
                      thread_index + i * width * height;
        _hessian_mat[g_index] = hessian[i];
    }
}

#define N 6

__device__ void matrix_inverse6x6(float (*a)[N], float (*b)[N])
{
    using namespace std;
    int i, j, k;
    float max, temp;
    // 定义一个临时矩阵t
    float t[N][N];
    // 将a矩阵临时存放在矩阵t[n][n]中
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            t[i][j] = a[i][j];
        }
    }
    // 初始化B矩阵为单位矩阵
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            b[i][j] = (i == j) ? (float)1 : 0;
        }
    }
    // 进行列主消元，找到每一列的主元
    for (i = 0; i < N; i++)
    {
        max = t[i][i];
        // 用于记录每一列中的第几个元素为主元
        k = i;
        // 寻找每一列中的主元元素
        for (j = i + 1; j < N; j++)
        {
            if (fabsf(t[j][i]) > fabsf(max))
            {
                max = t[j][i];
                k = j;
            }
        }
        // 如果主元所在的行不是第i行，则进行行交换
        if (k != i)
        {
            // 进行行交换
            for (j = 0; j < N; j++)
            {
                temp = t[i][j];
                t[i][j] = t[k][j];
                t[k][j] = temp;
                // 伴随矩阵B也要进行行交换
                temp = b[i][j];
                b[i][j] = b[k][j];
                b[k][j] = temp;
            }
        }
        if (t[i][i] == 0)
        {
            break;
        }
        // 获取列主元素
        temp = t[i][i];
        for (j = 0; j < N; j++)
        {
            t[i][j] = t[i][j] / temp;
            b[i][j] = b[i][j] / temp;
        }
        for (j = 0; j < N; j++)
        {
            if (j != i)
            {
                temp = t[j][i];
                // 消去该列的其他元素
                for (k = 0; k < N; k++)
                {
                    t[j][k] = t[j][k] - temp * t[i][k];
                    b[j][k] = b[j][k] - temp * b[i][k];
                }
            }
        }
    }
}
__device__ void mul(float A[N][N], float B[N][N], float C[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // 若绝对值小于10^-10,则置为0（这是我自己定的）
    for (int i = 0; i < N * N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (abs(C[i][j]) < pow(10, -6))
            {
                C[i][j] = 0;
            }
        }
    }
}
__device__ void inverse3x3(float *a, float (*inv)[3])
{
    float det =  a[0 * 3 + 0] * (a[1 * 3 + 1] * a[2 * 3 + 2] - a[2 * 3 + 1] * a[1 * 3 + 2]) -
                 a[1 * 3 + 0] * (a[0 * 3 + 1] * a[2 * 3 + 2] - a[2 * 3 + 1] * a[0 * 3 + 2]) +
                 a[2 * 3 + 0] * (a[0 * 3 + 1] * a[1 * 3 + 2] - a[1 * 3 + 1] * a[0 * 3 + 2]);
    if (det != 0)
    {
        inv[0][0] = (a[1 * 3 + 1] * a[2 * 3 + 2] - a[2 * 3 + 1] * a[1 * 3 + 2]) / det;
        inv[0][1] = (a[2 * 3 + 1] * a[0 * 3 + 2] - a[0 * 3 + 1] * a[2 * 3 + 2]) / det;
        inv[0][2] = (a[0 * 3 + 1] * a[1 * 3 + 2] - a[1 * 3 + 1] * a[0 * 3 + 2]) / det;
        inv[1][0] = (a[2 * 3 + 0] * a[1 * 3 + 2] - a[1 * 3 + 0] * a[2 * 3 + 2]) / det;
        inv[1][1] = (a[0 * 3 + 0] * a[2 * 3 + 2] - a[2 * 3 + 0] * a[0 * 3 + 2]) / det;
        inv[1][2] = (a[1 * 3 + 0] * a[0 * 3 + 2] - a[0 * 3 + 0] * a[1 * 3 + 2]) / det;
        inv[2][0] = (a[1 * 3 + 0] * a[2 * 3 + 1] - a[2 * 3 + 0] * a[1 * 3 + 1]) / det;
        inv[2][1] = (a[2 * 3 + 0] * a[0 * 3 + 1] - a[0 * 3 + 0] * a[2 * 3 + 1]) / det;
        inv[2][2] = (a[0 * 3 + 0] * a[1 * 3 + 1] - a[1 * 3 + 0] * a[0 * 3 + 1]) / det;
    }else
    {
        inv[0][0] = 1;
        inv[0][1] = 0;
        inv[0][2] = 0;
        inv[1][0] = 0;
        inv[1][1] = 1;
        inv[1][2] = 0;
        inv[2][0] = 0;
        inv[2][1] = 0;
        inv[2][2] = 1;
    }
}

__global__ void calMeanAndSigma(int subset, int sideW, int width, int height, uchar *_origin_src_image,
                                float *_mean_image, float *_sigma_image)
{
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % 32;
    int thread_y = thread_index / 32;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * 32;
    int g_y = blockIdx.y * 32;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);
    __shared__ uchar _src_image_sm[32 * 32]; // 1k

    for (int i = 0; i < (32 / ((BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y / 32))); ++i)
    {
        _src_image_sm[(thread_y + i * 2) * 32 + thread_x] =
            _origin_src_image[(g_y + thread_y + i * 2) * width + g_x + thread_x];
    }
    __syncthreads();

    const int lane_id = thread_index % 8;

    float sum = 0.0f;
    float buffer = 0;
    float sum_squre = 0.0f;
    for (int j = -halfSubset; j <= halfSubset; j++)
    {
        for (int k = -(halfSubset + lane_id); k < (halfSubset + 8 - lane_id); k++)
        {
            if (lane_id == 0)
            {
                buffer = _src_image_sm[(halfSubset + threadIdx.y + j) * 32 +
                                       halfSubset + k + threadIdx.x];
            }
            buffer = __shfl_sync(0xFFFFFFFFU, buffer, 0, 8);
            if (k >= -7 && k <= 7)
            {
                sum += buffer;
                sum_squre += buffer * buffer;
            }
        }
    }
    float mean_value = float(sum) / float(subset * subset);
    float sigma = float(sum_squre) - mean_value * float(sum);
    sigma = sqrt(sigma);
    _sigma_image[(g_y + halfWinSize + threadIdx.y) * width + g_x + halfWinSize + threadIdx.x] = sigma;
    _mean_image[(g_y + halfWinSize + threadIdx.y) * width + g_x + halfWinSize + threadIdx.x] = mean_value;
}

__global__ void calMeanImage(int subset, int sideW, int width, int height, uchar *_src_image, float *_mean_image)
{
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % BLOCK_DATA_DIM_X;
    int thread_y = thread_index / BLOCK_DATA_DIM_X;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * BLOCK_DATA_DIM_X;
    int g_y = blockIdx.y * BLOCK_DATA_DIM_Y;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);
    __shared__ uchar _src_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y];  // 4k
    __shared__ float _mean_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k

    // for (int i = 0; i < NUM_PER_THREAD_Y; i++)
    // {
    //     uchar value = _origin_src_image[(g_y + thread_y + i * 8) * width + g_x + thread_x];
    //     _src_image_sm[(thread_y + i * 8) * BLOCK_DATA_DIM_X + thread_x] = value;
    //     _mean_image_sm[(thread_y + i * 8) * BLOCK_DATA_DIM_X + thread_x] = value;
    // }
    __syncthreads();

    float sum = 0.0f;
    for (int j = -halfSubset; j <= halfSubset; j++) // y
    {
        for (int k = -halfSubset; k <= halfSubset; k++) // x
        {
            uchar value = _src_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                        halfWinSize + j + threadIdx.x];
            sum += value;
        }
    }
    float mean_value = float(sum) / float(subset * subset);
    __syncthreads();
    // int block_thread_index = thread_index / 4;
    // int block_thread_x = block_thread_index % 8;
    // int block_thread_y = block_thread_index / 8;

    // int g_sub_thread_index = thread_index % 4;
    // int g_block_thread_index = block_thread_index * 4 + sub_thread_index_x;
    // int sub_thread_x = g_sub_thread_index % 2;
    // int sub_thread_y = g_sub_thread_index / 2;
    // int num_x = halfSubset + 1;
    // int num_y = halfSubset + 1;
    // for (int r = num_y / 2; r > 0; r >> 1)
    // {
    //     for (int c = num_x / 2; c > 0; c >> 1)
    //     {
    //         for (int iRow = 0; iRow < r / 2; iRow++)
    //         {
    //             for (int iCol = 0; iCol < c / 2; iCol++)
    //             {
    //                 _mean_image_sm[(halfWinSize + block_thread_y + sub_thread_y + iRow) * 32 + halfWinSize + block_thread_x + sub_thread_x + iCol]
    //                     += _src_image_sm[(halfWinSize + block_thread_y + + sub_thread_y + iRow + r) * 32 + halfWinSize + block_thread_x + sub_thread_x + iCol + c];

    //                 _mean_image_sm[(halfWinSize + block_thread_y + sub_thread_y + iRow) * 32 + halfWinSize + block_thread_x + sub_thread_x + iCol]
    //                     += _src_image_sm[(halfWinSize + block_thread_y + + sub_thread_y + iRow + r) * 32 + halfWinSize + block_thread_x + sub_thread_x + iCol - c];

    //                 _mean_image_sm[(halfWinSize + block_thread_y + sub_thread_y + iRow) * 32 + halfWinSize + block_thread_x + sub_thread_x + iCol]
    //                     += _src_image_sm[(halfWinSize + block_thread_y + + sub_thread_y + iRow - r) * 32 + halfWinSize + block_thread_x + sub_thread_x + iCol + c];

    //                 _mean_image_sm[(halfWinSize + block_thread_y + sub_thread_y + iRow) * 32 + halfWinSize + block_thread_x + sub_thread_x + iCol]
    //                     += _src_image_sm[(halfWinSize + block_thread_y + + sub_thread_y + iRow - r) * 32 + halfWinSize + block_thread_x + sub_thread_x + iCol - c];
    //             }

    //         }
    //     }
    // }
}

__device__ void calTargetImageSubRegion(int subset, int sideW, int maxIterNum, uchar *_origin_image_target,
                                        float (*warP)[3], float *_target_value_intp, float *_sum_target_intp)
{

    // float MBT[4][4] = {{-0.166666666666667, 0.5, -0.5, 0.166666666666667},
    //                    {0.5, -1, 0, 0.666666666666667},
    //                    {-0.5, 0.5, 0.5, 0.166666666666667},
    //                    {0.166666666666667, 0, 0, 0}};
    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int n = 0;
    for (int j = -halfSubset; j <= halfSubset; j++)
    {
        for (int k = -halfSubset; k <= halfSubset; k++)
        {
            float x = warP[0][0] * k + warP[0][1] * j + warP[0][2] * 1 + halfWinSize;
            float y = warP[1][0] * k + warP[1][1] * j + warP[1][2] * 1 + halfWinSize;
            float z = warP[2][0] * k + warP[2][1] * j + warP[2][2] * 1 + halfWinSize;
            int x_int = floor(x);
            int y_int = floor(y);
            int z_int = floor(z);
            float delta_x = x - x_int;
            float delta_y = y - y_int;
            float delta_z = z - z_int;
            float weight_x[4] = {0};
            weight_x[0] = MBT[0][0] * x * x * x + MBT[0][1] * x * x + MBT[0][2] * x + MBT[0][3];
            weight_x[1] = MBT[1][0] * x * x * x + MBT[1][1] * x * x + MBT[1][2] * x + MBT[1][3];
            weight_x[2] = MBT[2][0] * x * x * x + MBT[2][1] * x * x + MBT[2][2] * x + MBT[2][3];
            weight_x[3] = MBT[3][0] * x * x * x + MBT[3][1] * x * x + MBT[3][2] * x + MBT[3][3];
            float weight_y[4] = {0};
            weight_y[0] = MBT[0][0] * y * y * y + MBT[0][1] * y * y + MBT[0][2] * y + MBT[0][3];
            weight_y[1] = MBT[1][0] * y * y * y + MBT[1][1] * y * y + MBT[1][2] * y + MBT[1][3];
            weight_y[2] = MBT[2][0] * y * y * y + MBT[2][1] * y * y + MBT[2][2] * y + MBT[2][3];
            weight_y[3] = MBT[3][0] * y * y * y + MBT[3][1] * y * y + MBT[3][2] * y + MBT[3][3];
            float target_value_intp[4][4] = {0};
            target_value_intp[0][0] = _origin_image_target[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + threadIdx.x];
            target_value_intp[1][0] = _origin_image_target[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + threadIdx.x];
            target_value_intp[2][0] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + threadIdx.x];
            target_value_intp[3][0] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + threadIdx.x];
            target_value_intp[0][1] = _origin_image_target[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 1 + threadIdx.x];
            target_value_intp[1][1] = _origin_image_target[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 1 + threadIdx.x];
            target_value_intp[2][1] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 1 + threadIdx.x];
            target_value_intp[3][1] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 1 + threadIdx.x];
            target_value_intp[0][2] = _origin_image_target[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 2 + threadIdx.x];
            target_value_intp[1][2] = _origin_image_target[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 2 + threadIdx.x];
            target_value_intp[2][2] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 2 + threadIdx.x];
            target_value_intp[3][2] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 2 + threadIdx.x];
            target_value_intp[0][3] = _origin_image_target[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 3 + threadIdx.x];
            target_value_intp[1][3] = _origin_image_target[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 3 + threadIdx.x];
            target_value_intp[2][3] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 3 + threadIdx.x];
            target_value_intp[3][3] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                           halfWinSize + x_int + 3 + threadIdx.x];

            _target_value_intp[n] += target_value_intp[0][0] * weight_y[0] * weight_x[0];
            _target_value_intp[n] += target_value_intp[1][0] * weight_y[1] * weight_x[0];
            _target_value_intp[n] += target_value_intp[2][0] * weight_y[2] * weight_x[0];
            _target_value_intp[n] += target_value_intp[3][0] * weight_y[3] * weight_x[0];
            _target_value_intp[n] += target_value_intp[0][1] * weight_y[0] * weight_x[1];
            _target_value_intp[n] += target_value_intp[1][1] * weight_y[1] * weight_x[1];
            _target_value_intp[n] += target_value_intp[2][1] * weight_y[2] * weight_x[1];
            _target_value_intp[n] += target_value_intp[3][1] * weight_y[3] * weight_x[1];
            _target_value_intp[n] += target_value_intp[0][2] * weight_y[0] * weight_x[2];
            _target_value_intp[n] += target_value_intp[1][2] * weight_y[1] * weight_x[2];
            _target_value_intp[n] += target_value_intp[2][2] * weight_y[2] * weight_x[2];
            _target_value_intp[n] += target_value_intp[3][2] * weight_y[3] * weight_x[2];
            _target_value_intp[n] += target_value_intp[0][3] * weight_y[0] * weight_x[3];
            _target_value_intp[n] += target_value_intp[1][3] * weight_y[1] * weight_x[3];
            _target_value_intp[n] += target_value_intp[2][3] * weight_y[2] * weight_x[3];
            _target_value_intp[n] += target_value_intp[3][3] * weight_y[3] * weight_x[3];
            _sum_target_intp[0] += _target_value_intp[n];
            n++;
        }
    }
}

__global__ void calInvHessian(int subset, int sideW, int width, int height, float *_hessian_image, float *_inv_hessian_image)
{
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % BLOCK_DATA_DIM_X;
    int thread_y = thread_index / BLOCK_DATA_DIM_X;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);

    __shared__ float _hessian_image_sm[BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y * 36]; // 9k
    int block_index = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = 0; i < 6 * 6; i++)
    {
        int g_index = (halfWinSize)*width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y +
                      thread_index + i * width * height;
        _hessian_image_sm[i * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y + thread_index] = _hessian_image[g_index];
    }

    __syncthreads();
    float hessian[6][6];
    // curandState states[36];
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            hessian[i][j] = _hessian_image_sm[(i * 6 + j) * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y + thread_index];
        }
    }

    float inv_hessian[6][6];
    matrix_inverse6x6(hessian, inv_hessian);
    __syncthreads();
    /*float e[6][6] = {0};
    mul(hessian,inv_hessian,e);

    if(thread_index == 0){
        for (int i = 0; i < 6; i++) {
            for (int j = 0;j < 6;j++) {

                //if(thread_index == 0)
                {
                    printf("e[%d][%d]: %f ",i, j, e[i][j]);
                }
            }
            //if(thread_index == 0)
            {
                printf("\n");
            }

        }
    }*/

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            int g_index = (halfWinSize)*width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y +
                          thread_index + (i * 6 + j) * width * height;
            _inv_hessian_image[g_index] = inv_hessian[i][j];
        }
    }
}

__global__ void calTargetMeanImageAndSigma(int subset, int sideW, int width, int height, uchar *_origin_target_image,float (*warP)[3], 
                                           float *_mean_image, float *_sigma_image)
{
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % 32;
    int thread_y = thread_index / 32;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * 32;
    int g_y = blockIdx.y * 32;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);
    __shared__ uchar _src_image_sm[32 * 32]; // 4k

    for (int i = 0; i < (32 / ((BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y / 32))); ++i)
    {
        _src_image_sm[(thread_y + i * 2) * 32 + thread_x] =
            _origin_target_image[(g_y + thread_y + i * 2) * width + g_x + thread_x];
    }
    __syncthreads();

    float MBT[4][4] = {{-0.166666666666667, 0.5, -0.5, 0.166666666666667},
                       {0.5, -1, 0, 0.666666666666667},
                       {-0.5, 0.5, 0.5, 0.166666666666667},
                       {0.166666666666667, 0, 0, 0}};
    const int lane_id = thread_index % 8;
    int n = 0;
    for (int j = -halfSubset; j <= halfSubset; j++)
    {
        for (int k = -(halfSubset + lane_id); k < (halfSubset + 8 - lane_id); k++)
        {
            float weight_x[4] = {0};
            float weight_y[4] = {0};
            int x_int = 0;
            int y_int = 0;
            int z_int = 0;
            if (lane_id == 0)
            {
                float x = warP[0][0] * k + warP[0][1] * j + warP[0][2] * 1 + halfWinSize;
                float y = warP[1][0] * k + warP[1][1] * j + warP[1][2] * 1 + halfWinSize;
                float z = warP[2][0] * k + warP[2][1] * j + warP[2][2] * 1 + halfWinSize;
                x_int = floor(x);
                y_int = floor(y);
                z_int = floor(z);
                float delta_x = x - x_int;
                float delta_y = y - y_int;
                float delta_z = z - z_int;
                
                weight_x[0] = MBT[0][0] * x * x * x + MBT[0][1] * x * x + MBT[0][2] * x + MBT[0][3];
                weight_x[1] = MBT[1][0] * x * x * x + MBT[1][1] * x * x + MBT[1][2] * x + MBT[1][3];
                weight_x[2] = MBT[2][0] * x * x * x + MBT[2][1] * x * x + MBT[2][2] * x + MBT[2][3];
                weight_x[3] = MBT[3][0] * x * x * x + MBT[3][1] * x * x + MBT[3][2] * x + MBT[3][3];
                
                weight_y[0] = MBT[0][0] * y * y * y + MBT[0][1] * y * y + MBT[0][2] * y + MBT[0][3];
                weight_y[1] = MBT[1][0] * y * y * y + MBT[1][1] * y * y + MBT[1][2] * y + MBT[1][3];
                weight_y[2] = MBT[2][0] * y * y * y + MBT[2][1] * y * y + MBT[2][2] * y + MBT[2][3];
                weight_y[3] = MBT[3][0] * y * y * y + MBT[3][1] * y * y + MBT[3][2] * y + MBT[3][3];
            }
            weight_x[0] = __shfl_sync(0xFFFFFFFFU, weight_x[0], 0, 8);
            weight_x[1] = __shfl_sync(0xFFFFFFFFU, weight_x[1], 0, 8);
            weight_x[2] = __shfl_sync(0xFFFFFFFFU, weight_x[2], 0, 8);
            weight_x[3] = __shfl_sync(0xFFFFFFFFU, weight_x[3], 0, 8);

            weight_y[0] = __shfl_sync(0xFFFFFFFFU, weight_y[0], 0, 8);
            weight_y[1] = __shfl_sync(0xFFFFFFFFU, weight_y[1], 0, 8);
            weight_y[2] = __shfl_sync(0xFFFFFFFFU, weight_y[2], 0, 8);
            weight_y[3] = __shfl_sync(0xFFFFFFFFU, weight_y[3], 0, 8);

            x_int = __shfl_sync(0xFFFFFFFFU, x_int, 0, 8);
            y_int = __shfl_sync(0xFFFFFFFFU, y_int, 0, 8);
            z_int = __shfl_sync(0xFFFFFFFFU, z_int, 0, 8);
            if (k >= -7 && k <= 7)
            {
                float target_value_intp[4][4] = {0};
                target_value_intp[0][0] = _origin_target_image[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + threadIdx.x];
                target_value_intp[1][0] = _origin_target_image[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + threadIdx.x];
                target_value_intp[2][0] = _origin_target_image[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + threadIdx.x];
                target_value_intp[3][0] = _origin_target_image[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + threadIdx.x];
                target_value_intp[0][1] = _origin_target_image[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 1 + threadIdx.x];
                target_value_intp[1][1] = _origin_target_image[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 1 + threadIdx.x];
                target_value_intp[2][1] = _origin_target_image[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 1 + threadIdx.x];
                target_value_intp[3][1] = _origin_target_image[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 1 + threadIdx.x];
                target_value_intp[0][2] = _origin_target_image[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 2 + threadIdx.x];
                target_value_intp[1][2] = _origin_target_image[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 2 + threadIdx.x];
                target_value_intp[2][2] = _origin_target_image[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 2 + threadIdx.x];
                target_value_intp[3][2] = _origin_target_image[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 2 + threadIdx.x];
                target_value_intp[0][3] = _origin_target_image[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 3 + threadIdx.x];
                target_value_intp[1][3] = _origin_target_image[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 3 + threadIdx.x];
                target_value_intp[2][3] = _origin_target_image[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 3 + threadIdx.x];
                target_value_intp[3][3] = _origin_target_image[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                               halfWinSize + x_int + 3 + threadIdx.x];

                // _target_value_intp[n] += target_value_intp[0][0] * weight_y[0] * weight_x[0];
                // _target_value_intp[n] += target_value_intp[1][0] * weight_y[1] * weight_x[0];
                // _target_value_intp[n] += target_value_intp[2][0] * weight_y[2] * weight_x[0];
                // _target_value_intp[n] += target_value_intp[3][0] * weight_y[3] * weight_x[0];
                // _target_value_intp[n] += target_value_intp[0][1] * weight_y[0] * weight_x[1];
                // _target_value_intp[n] += target_value_intp[1][1] * weight_y[1] * weight_x[1];
                // _target_value_intp[n] += target_value_intp[2][1] * weight_y[2] * weight_x[1];
                // _target_value_intp[n] += target_value_intp[3][1] * weight_y[3] * weight_x[1];
                // _target_value_intp[n] += target_value_intp[0][2] * weight_y[0] * weight_x[2];
                // _target_value_intp[n] += target_value_intp[1][2] * weight_y[1] * weight_x[2];
                // _target_value_intp[n] += target_value_intp[2][2] * weight_y[2] * weight_x[2];
                // _target_value_intp[n] += target_value_intp[3][2] * weight_y[3] * weight_x[2];
                // _target_value_intp[n] += target_value_intp[0][3] * weight_y[0] * weight_x[3];
                // _target_value_intp[n] += target_value_intp[1][3] * weight_y[1] * weight_x[3];
                // _target_value_intp[n] += target_value_intp[2][3] * weight_y[2] * weight_x[3];
                // _target_value_intp[n] += target_value_intp[3][3] * weight_y[3] * weight_x[3];
                n++;
            }
        }
    }
    // float mean_value = float(sum) / float(subset * subset);
    // float sigma = float(sum_squre) - mean_value * float(sum);
    // sigma = sqrt(sigma);
    // _sigma_image[(g_y + halfWinSize + threadIdx.y) * width + g_x + halfWinSize + threadIdx.x] = sigma;
    // _mean_image[(g_y + halfWinSize + threadIdx.y) * width + g_x + halfWinSize + threadIdx.x] = mean_value;
}

__global__ void calOptDisp_singleThreadBlock(int subset, int sideW, int maxIterNum, int width, int height,
                                             float *_init_p, uchar *_origin_image_ref, uchar *_origin_image_target,
                                             float *_x_grad_image, float *_y_grad_image, float *_mean_image, float *_sigma_image_gpu,
                                             float *_hessian_inv_image, float *_origin_disp,
                                             float *_opt_disp)
{
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    
    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X;
    int g_y = blockIdx.y * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_Y;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);


    __syncthreads();
    __shared__ float _target_value_intp_sm[16 * 16];//1k
    __shared__ float _target_sigma_value_intp_sm[16 * 16];//1k;
    __shared__ float _target_delta_mean_value_intp_sm[16 * 16];//1k
    __shared__ float _hessian_inv_image_sm[BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_X * 36]; // 9k;
    __shared__ float deltap_sm[6 * 16 * 16];
    int block_index = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = 0; i < 6 * 6; i++)
    {
        int g_index = (halfWinSize)*width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y +
                      thread_index + i * width * height;
        if(thread_index < BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y){
            _hessian_inv_image_sm[i * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y + thread_index] = _hessian_inv_image[g_index];
        }
    }
    __syncthreads();
    
   
    int thread_x = thread_index % 16;
    int thread_y = thread_index / 16;
    for (int i = 0; i < BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y; i++)
    {
        float warP[3][3] = {{1 + _init_p[1], _init_p[2], _init_p[0]},
                            {_init_p[4], 1 + _init_p[5], _init_p[3]},
                            {0, 0, 1}};
        float _target_value_intp = 0;
        if (thread_x <= subset && thread_y <= subset)
        {
            int iRow = i / BLOCK_THREAD_DIM_X + g_y + halfWinSize;
            int iCol = i % BLOCK_THREAD_DIM_X + g_x + halfWinSize;
            if(i == 0 && thread_x == 0 && thread_y == 0 && block_index == 0){
                printf("i: %d, iRow: %d, iCol: %d\n",i, iRow, iCol);
            }
            float thre = 1;
            int Iter = 0;
            float Czncc = 0;
            while (thre > 1e-3 && Iter < maxIterNum || Iter == 0)
            {

                float weight_x[4] = {0};
                float weight_y[4] = {0};
                int x_int = 0;
                int y_int = 0;
                int z_int = 0;
                int coor_x = thread_x - halfSubset;
                int coor_y = thread_y - halfSubset;
                float x = warP[0][0] * coor_x + warP[0][1] * coor_y + warP[0][2] * 1;
                float y = warP[1][0] * coor_x + warP[1][1] * coor_y + warP[1][2] * 1;
                float z = warP[2][0] * coor_x + warP[2][1] * coor_y + warP[2][2] * 1;

                x_int = floor(x);
                y_int = floor(y);
                z_int = floor(z);
                // if(threadIdx.y == 0 && threadIdx.x == 0){
                //     printf("iRow: %d,iCol: %d,i:%d, x: %lf, y: %lf, z: %lf, coor_x: %d, coor_y: %d, x_int: %d, y_int: %d,z_int: %d\n",
                //             iRow, iCol, i,x, y, z, coor_x, coor_y, x_int, y_int, z_int);
                //     // printf("warP[0][0]: %lf,warP[0][1]: %lf, warP[0][2]: %lf, warP[0][0]: %lf,warP[0][1]: %lf, warP[0][2]: %lf,warP[0][0]: %lf,warP[0][1]: %lf, warP[0][2]: %lf\n",
                //     //         warP[0][0], warP[0][1], warP[0][2],
                //     //         warP[1][0], warP[1][1], warP[1][2],
                //     //         warP[2][0], warP[2][1], warP[2][2]);
                // }
                float delta_x = x - x_int;
                float delta_y = y - y_int;
                float delta_z = z - z_int;

                weight_x[0] = MBT[0][0] * x * x * x + MBT[0][1] * x * x + MBT[0][2] * x + MBT[0][3];
                weight_x[1] = MBT[1][0] * x * x * x + MBT[1][1] * x * x + MBT[1][2] * x + MBT[1][3];
                weight_x[2] = MBT[2][0] * x * x * x + MBT[2][1] * x * x + MBT[2][2] * x + MBT[2][3];
                weight_x[3] = MBT[3][0] * x * x * x + MBT[3][1] * x * x + MBT[3][2] * x + MBT[3][3];

                weight_y[0] = MBT[0][0] * y * y * y + MBT[0][1] * y * y + MBT[0][2] * y + MBT[0][3];
                weight_y[1] = MBT[1][0] * y * y * y + MBT[1][1] * y * y + MBT[1][2] * y + MBT[1][3];
                weight_y[2] = MBT[2][0] * y * y * y + MBT[2][1] * y * y + MBT[2][2] * y + MBT[2][3];
                weight_y[3] = MBT[3][0] * y * y * y + MBT[3][1] * y * y + MBT[3][2] * y + MBT[3][3];

                float target_value_intp[4][4] = {0};
                target_value_intp[0][0] = _origin_image_target[(iRow + y_int - 1) * width + x_int + iCol];
                target_value_intp[1][0] = _origin_image_target[(iRow + y_int) * width + x_int + iCol];
                target_value_intp[2][0] = _origin_image_target[(iRow + y_int + 1) * width + x_int + iCol];
                target_value_intp[3][0] = _origin_image_target[(iRow + y_int + 2) * width + x_int + iCol];
                target_value_intp[0][1] = _origin_image_target[(iRow + y_int - 1) * width + x_int + 1 + iCol];
                target_value_intp[1][1] = _origin_image_target[(iRow + y_int) * width + x_int + 1 + iCol];
                target_value_intp[2][1] = _origin_image_target[(iRow + y_int + 1) * width + x_int + 1 + iCol];
                target_value_intp[3][1] = _origin_image_target[(iRow + y_int + 2) * width + x_int + 1 + iCol];
                target_value_intp[0][2] = _origin_image_target[(iRow + y_int - 1) * width + x_int + 2 + iCol];
                target_value_intp[1][2] = _origin_image_target[(iRow + y_int) * width + x_int + 2 + iCol];
                target_value_intp[2][2] = _origin_image_target[(iRow + y_int + 1) * width + x_int + 2 + iCol];
                target_value_intp[3][2] = _origin_image_target[(iRow + y_int + 2) * width + x_int + 2 + iCol];
                target_value_intp[0][3] = _origin_image_target[(iRow + y_int - 1) * width + x_int + 3 + iCol];
                target_value_intp[1][3] = _origin_image_target[(iRow + y_int) * width + x_int + 3 + iCol];
                target_value_intp[2][3] = _origin_image_target[(iRow + y_int + 1) * width + x_int + 3 + iCol];
                target_value_intp[3][3] = _origin_image_target[(iRow + y_int + 2) * width + x_int + 3 + iCol];

                // int tmp_index = (iRow + y_int + 2) * width + x_int + 3 + iCol;
                // if((iRow + y_int + 2) > height || (x_int + 3 + iCol) > width){
                //     printf("i: %d, iRow: %d, iCol: %d, g_y: %d,g_x: %d\n",i, (iRow + y_int + 2), (x_int + 3 + iCol),g_y, g_x);
                // }

                _target_value_intp += target_value_intp[0][0] * weight_y[0] * weight_x[0];
                _target_value_intp += target_value_intp[1][0] * weight_y[1] * weight_x[0];
                _target_value_intp += target_value_intp[2][0] * weight_y[2] * weight_x[0];
                _target_value_intp += target_value_intp[3][0] * weight_y[3] * weight_x[0];
                _target_value_intp += target_value_intp[0][1] * weight_y[0] * weight_x[1];
                _target_value_intp += target_value_intp[1][1] * weight_y[1] * weight_x[1];
                _target_value_intp += target_value_intp[2][1] * weight_y[2] * weight_x[1];
                _target_value_intp += target_value_intp[3][1] * weight_y[3] * weight_x[1];
                _target_value_intp += target_value_intp[0][2] * weight_y[0] * weight_x[2];
                _target_value_intp += target_value_intp[1][2] * weight_y[1] * weight_x[2];
                _target_value_intp += target_value_intp[2][2] * weight_y[2] * weight_x[2];
                _target_value_intp += target_value_intp[3][2] * weight_y[3] * weight_x[2];
                _target_value_intp += target_value_intp[0][3] * weight_y[0] * weight_x[3];
                _target_value_intp += target_value_intp[1][3] * weight_y[1] * weight_x[3];
                _target_value_intp += target_value_intp[2][3] * weight_y[2] * weight_x[3];
                _target_value_intp += target_value_intp[3][3] * weight_y[3] * weight_x[3];

                _target_value_intp_sm[thread_y * 16 + thread_x] = _target_value_intp;
                _target_sigma_value_intp_sm[thread_y * 16 + thread_x] = _target_value_intp * _target_value_intp;
                _target_delta_mean_value_intp_sm[thread_y * 16 + thread_x] = _target_value_intp;
                __syncthreads();

                for (int r = 16 / 2; r > 0; r >>= 1)
                {
                    for (int c = 16 / 2; c > 0; c >>= 1)
                    {
                        if (thread_y < r && thread_x < c)
                        {
                            _target_value_intp_sm[thread_y * 16 + thread_x] += _target_value_intp_sm[(thread_y + r) * 16 + (thread_x + c)];
                            _target_sigma_value_intp_sm[thread_y * 16 + thread_x] += _target_sigma_value_intp_sm[(thread_y + r) * 16 + (thread_x + c)];
                        }
                        __syncthreads();
                    }
                }
                __syncthreads();
                float target_value_sum = _target_value_intp_sm[0];
                float target_mean_value = target_value_sum / float(subset * subset);
                float target_delta_squar = _target_sigma_value_intp_sm[0] - target_value_sum * target_mean_value;
                target_delta_squar = sqrt(target_delta_squar);

                _target_delta_mean_value_intp_sm[thread_y * 16 + thread_x] -= target_mean_value;
                __syncthreads();

                int index = (thread_y - halfSubset + iRow) * width + thread_x - halfSubset + iCol;
                int sub_region_index = (iRow)*width + iCol;

                uchar image_value_ref = _origin_image_ref[index];

                float mean_image_value_ref = _mean_image[sub_region_index];
                float sigma_image_value_ref = _sigma_image_gpu[sub_region_index];

                float ref_delta_mean_value = image_value_ref - mean_image_value_ref;

                float Jacobian[6] = {0};
                Jacobian[0] = _x_grad_image[index];
                Jacobian[3] = _y_grad_image[index];
                float coord_weight[2] = {0};
                coord_weight[0] = float(thread_y - halfSubset) / float(halfSubset + 1);
                coord_weight[1] = float(thread_x - halfSubset) / float(halfSubset + 1);

                Jacobian[1] = Jacobian[0] * coord_weight[0]; // x;
                Jacobian[2] = Jacobian[0] * coord_weight[1]; // y;
                Jacobian[4] = Jacobian[3] * coord_weight[0];
                Jacobian[5] = Jacobian[3] * coord_weight[1];

                float invHJacobian[6] = {0};
                for (int jRow = 0; jRow < 6; jRow++)
                {
                    for (int jCol = 0; jCol < 6; jCol++)
                    {
                        int g_index_sm = jRow * 6 + jCol;
                        int tmp_index = g_index_sm * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y + i;
                        invHJacobian[jCol] += _hessian_inv_image_sm[tmp_index] * Jacobian[jCol];
                        // if (i == 0 && thread_x == 0 && thread_y == 0 && block_index == 40)
                        // {
                        //     printf("i: %d, iRow: %d, iCol: %d,_hessian_inv_image_sm[%d]: %lf\n",
                        //             i, iRow, iCol,tmp_index, _hessian_inv_image_sm[tmp_index]);
                        // }
                    }
                }

                if (i == 0 && thread_x == 0 && thread_y == 14 && block_index == 40)
                {
                    for (int m = 0; m < 6; m++)
                    {
                        printf("i: %d, iRow: %d, iCol: %d, invHJacobian[%d]: %lf,Jacobian[%d]: %lf\n",
                           i, iRow, iCol,  m,invHJacobian[m], m, Jacobian[m]);
                    }
                }
                __syncthreads();
                deltap_sm[thread_index + 0 * 16 * 16] = 0.0f;
                deltap_sm[thread_index + 1 * 16 * 16] = 0.0f;
                deltap_sm[thread_index + 2 * 16 * 16] = 0.0f;
                deltap_sm[thread_index + 3 * 16 * 16] = 0.0f;
                deltap_sm[thread_index + 4 * 16 * 16] = 0.0f;
                deltap_sm[thread_index + 5 * 16 * 16] = 0.0f;

                __syncthreads();
                float target_delta_mean_value = _target_delta_mean_value_intp_sm[thread_y * 16 + thread_x];
                float tmp = 0;
                if (target_delta_mean_value != 0 && target_delta_squar != 0)
                {
                    tmp = ref_delta_mean_value - sigma_image_value_ref / target_delta_squar * target_delta_mean_value;
                }
                float deltap[6] = {0};
                deltap[0] = -invHJacobian[0] * tmp;
                deltap[1] = -invHJacobian[1] * tmp;
                deltap[2] = -invHJacobian[2] * tmp;
                deltap[3] = -invHJacobian[3] * tmp;
                deltap[4] = -invHJacobian[4] * tmp;
                deltap[5] = -invHJacobian[5] * tmp;

                float M[6] = {1, 1.0f / subset, 1.0f / subset, 1, 1.0f / subset, 1.0f / subset};
                deltap_sm[thread_index + 0 * 16 * 16] = M[0] * deltap[0];
                deltap_sm[thread_index + 1 * 16 * 16] = M[1] * deltap[1];
                deltap_sm[thread_index + 2 * 16 * 16] = M[2] * deltap[2];
                deltap_sm[thread_index + 3 * 16 * 16] = M[3] * deltap[3];
                deltap_sm[thread_index + 4 * 16 * 16] = M[4] * deltap[4];
                deltap_sm[thread_index + 5 * 16 * 16] = M[5] * deltap[5];

                __syncthreads();

                for (int th_index = 16 * 16 / 2; th_index > 0; th_index >>= 1)
                {
                    if (thread_index < th_index)
                    {
                        deltap_sm[thread_index + 0 * 16 * 16] += deltap_sm[thread_index + th_index + 0 * 16 * 16];
                        deltap_sm[thread_index + 1 * 16 * 16] += deltap_sm[thread_index + th_index + 1 * 16 * 16];
                        deltap_sm[thread_index + 2 * 16 * 16] += deltap_sm[thread_index + th_index + 2 * 16 * 16];
                        deltap_sm[thread_index + 3 * 16 * 16] += deltap_sm[thread_index + th_index + 3 * 16 * 16];
                        deltap_sm[thread_index + 4 * 16 * 16] += deltap_sm[thread_index + th_index + 4 * 16 * 16];
                        deltap_sm[thread_index + 5 * 16 * 16] += deltap_sm[thread_index + th_index + 5 * 16 * 16];
                    }
                    __syncthreads();
                }

                float delta_warp_p[3 * 3] = {1 + deltap_sm[1 * 16 * 16], deltap_sm[2 * 16 * 16], deltap_sm[0 * 16 * 16],
                                             deltap_sm[4 * 16 * 16], 1 + deltap_sm[5 * 16 * 16], deltap_sm[3 * 16 * 16],
                                             0, 0, 1};
                float invwarpdelta[3][3] = {0};
                inverse3x3(delta_warp_p, invwarpdelta);
                float warP2[3][3] = {0};
                warP2[0][0] = warP[0][0] * invwarpdelta[0][0] + warP[0][1] * invwarpdelta[1][0] + warP[0][2] * invwarpdelta[2][0];
                warP2[0][1] = warP[0][0] * invwarpdelta[0][1] + warP[0][1] * invwarpdelta[1][1] + warP[0][2] * invwarpdelta[2][1];
                warP2[0][2] = warP[0][0] * invwarpdelta[0][2] + warP[0][1] * invwarpdelta[1][2] + warP[0][2] * invwarpdelta[2][2];
                warP2[1][0] = warP[1][0] * invwarpdelta[0][0] + warP[1][1] * invwarpdelta[1][0] + warP[1][2] * invwarpdelta[2][0];
                warP2[1][1] = warP[1][0] * invwarpdelta[0][1] + warP[1][1] * invwarpdelta[1][1] + warP[1][2] * invwarpdelta[2][1];
                warP2[1][2] = warP[1][0] * invwarpdelta[0][2] + warP[1][1] * invwarpdelta[1][2] + warP[1][2] * invwarpdelta[2][2];
                warP2[2][0] = warP[2][0] * invwarpdelta[0][0] + warP[2][1] * invwarpdelta[1][0] + warP[2][2] * invwarpdelta[2][0];
                warP2[2][1] = warP[2][0] * invwarpdelta[0][1] + warP[2][1] * invwarpdelta[1][1] + warP[2][2] * invwarpdelta[2][1];
                warP2[2][2] = warP[2][0] * invwarpdelta[0][2] + warP[2][1] * invwarpdelta[1][2] + warP[2][2] * invwarpdelta[2][2];

                // warP[0][0] = warP2[0][0]; warP[0][1] = warP2[0][1]; warP[0][2] = warP2[0][2];
                // warP[1][0] = warP2[1][0]; warP[1][1] = warP2[1][1]; warP[1][2] = warP2[1][2];
                // warP[2][0] = warP2[2][0]; warP[2][1] = warP2[2][1]; warP[2][2] = warP2[2][2];

                float delta_value = deltap_sm[0 * 16 * 16] * deltap_sm[0 * 16 * 16] +
                                    deltap_sm[3 * 16 * 16] * deltap_sm[3 * 16 * 16];
                thre = sqrt(delta_value);
                Iter++;
            }
            __syncthreads();
            int disp_index = (iRow) * width + iCol;
            _opt_disp[disp_index] = _origin_disp[disp_index] + warP[1][2];
        }

        __syncthreads();
    }
}

__global__ void calOptDisp(int subset, int sideW, int maxIterNum, int width, int height,
                           float *_init_p, uchar *_origin_image_ref, uchar *_origin_image_target,
                           float *_x_grad_image, float *_y_grad_image, float *_mean_image, float *_sigma_image_gpu,
                           float *_hessian_inv_image, float *_origin_disp,
                           float *_opt_disp)
{
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % BLOCK_DATA_DIM_X;
    int thread_y = thread_index / BLOCK_DATA_DIM_X;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);

    //__shared__ float _target_delta_intp_sm[subset * subset];
    __shared__ float _hessian_inv_image_sm[BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_X * 36]; // 9k;
    int block_index = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = 0; i < 6 * 6; i++)
    {
        int g_index = (halfWinSize)*width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y +
                      thread_index + i * width * height;
        _hessian_inv_image_sm[i * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y + thread_index] = _hessian_inv_image[g_index];
    }
    __syncthreads();
    float warP[3][3] = {{1 + _init_p[1], _init_p[2], _init_p[0]},
                        {_init_p[4], 1 + _init_p[5], _init_p[3]},
                        {0, 0, 1}};
    const int lane_id = thread_index % 8;
    int n = 0;
    float thre = 0.0f;
    int Iter = 0;
    while (thre > 1e-3 && Iter < maxIterNum || Iter == 0)
    {
        float _mean_target_intp = 0.0f;
        float _sum_target_intp = 0.0f;
        for (int j = -halfSubset; j <= halfSubset; j++)
        {
            for (int k = -(halfSubset + lane_id); k < (halfSubset + 8 - lane_id); k++)
            {
                float weight_x[4] = {0};
                float weight_y[4] = {0};
                int x_int = 0;
                int y_int = 0;
                int z_int = 0;
                if (lane_id == 0)
                {
                    float x = warP[0][0] * k + warP[0][1] * j + warP[0][2] * 1 + halfWinSize;
                    float y = warP[1][0] * k + warP[1][1] * j + warP[1][2] * 1 + halfWinSize;
                    float z = warP[2][0] * k + warP[2][1] * j + warP[2][2] * 1 + halfWinSize;
                    x_int = floor(x);
                    y_int = floor(y);
                    z_int = floor(z);
                    float delta_x = x - x_int;
                    float delta_y = y - y_int;
                    float delta_z = z - z_int;

                    weight_x[0] = MBT[0][0] * x * x * x + MBT[0][1] * x * x + MBT[0][2] * x + MBT[0][3];
                    weight_x[1] = MBT[1][0] * x * x * x + MBT[1][1] * x * x + MBT[1][2] * x + MBT[1][3];
                    weight_x[2] = MBT[2][0] * x * x * x + MBT[2][1] * x * x + MBT[2][2] * x + MBT[2][3];
                    weight_x[3] = MBT[3][0] * x * x * x + MBT[3][1] * x * x + MBT[3][2] * x + MBT[3][3];

                    weight_y[0] = MBT[0][0] * y * y * y + MBT[0][1] * y * y + MBT[0][2] * y + MBT[0][3];
                    weight_y[1] = MBT[1][0] * y * y * y + MBT[1][1] * y * y + MBT[1][2] * y + MBT[1][3];
                    weight_y[2] = MBT[2][0] * y * y * y + MBT[2][1] * y * y + MBT[2][2] * y + MBT[2][3];
                    weight_y[3] = MBT[3][0] * y * y * y + MBT[3][1] * y * y + MBT[3][2] * y + MBT[3][3];
                }
                weight_x[0] = __shfl_sync(0xFFFFFFFFU, weight_x[0], 0, 8);
                weight_x[1] = __shfl_sync(0xFFFFFFFFU, weight_x[1], 0, 8);
                weight_x[2] = __shfl_sync(0xFFFFFFFFU, weight_x[2], 0, 8);
                weight_x[3] = __shfl_sync(0xFFFFFFFFU, weight_x[3], 0, 8);

                weight_y[0] = __shfl_sync(0xFFFFFFFFU, weight_y[0], 0, 8);
                weight_y[1] = __shfl_sync(0xFFFFFFFFU, weight_y[1], 0, 8);
                weight_y[2] = __shfl_sync(0xFFFFFFFFU, weight_y[2], 0, 8);
                weight_y[3] = __shfl_sync(0xFFFFFFFFU, weight_y[3], 0, 8);

                x_int = __shfl_sync(0xFFFFFFFFU, x_int, 0, 8);
                y_int = __shfl_sync(0xFFFFFFFFU, y_int, 0, 8);
                z_int = __shfl_sync(0xFFFFFFFFU, z_int, 0, 8);
                if (k >= -7 && k <= 7)
                {
                    float target_value_intp[4][4] = {0};
                    target_value_intp[0][0] = _origin_image_target[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + threadIdx.x];
                    target_value_intp[1][0] = _origin_image_target[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + threadIdx.x];
                    target_value_intp[2][0] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + threadIdx.x];
                    target_value_intp[3][0] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + threadIdx.x];
                    target_value_intp[0][1] = _origin_image_target[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 1 + threadIdx.x];
                    target_value_intp[1][1] = _origin_image_target[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 1 + threadIdx.x];
                    target_value_intp[2][1] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 1 + threadIdx.x];
                    target_value_intp[3][1] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 1 + threadIdx.x];
                    target_value_intp[0][2] = _origin_image_target[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 2 + threadIdx.x];
                    target_value_intp[1][2] = _origin_image_target[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 2 + threadIdx.x];
                    target_value_intp[2][2] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 2 + threadIdx.x];
                    target_value_intp[3][2] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 2 + threadIdx.x];
                    target_value_intp[0][3] = _origin_image_target[(halfWinSize + threadIdx.y + y_int - 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 3 + threadIdx.x];
                    target_value_intp[1][3] = _origin_image_target[(halfWinSize + threadIdx.y + y_int) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 3 + threadIdx.x];
                    target_value_intp[2][3] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 1) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 3 + threadIdx.x];
                    target_value_intp[3][3] = _origin_image_target[(halfWinSize + threadIdx.y + y_int + 2) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                                   halfWinSize + x_int + 3 + threadIdx.x];
                    float _target_value_intp = 0;
                    _target_value_intp += target_value_intp[0][0] * weight_y[0] * weight_x[0];
                    _target_value_intp += target_value_intp[1][0] * weight_y[1] * weight_x[0];
                    _target_value_intp += target_value_intp[2][0] * weight_y[2] * weight_x[0];
                    _target_value_intp += target_value_intp[3][0] * weight_y[3] * weight_x[0];
                    _target_value_intp += target_value_intp[0][1] * weight_y[0] * weight_x[1];
                    _target_value_intp += target_value_intp[1][1] * weight_y[1] * weight_x[1];
                    _target_value_intp += target_value_intp[2][1] * weight_y[2] * weight_x[1];
                    _target_value_intp += target_value_intp[3][1] * weight_y[3] * weight_x[1];
                    _target_value_intp += target_value_intp[0][2] * weight_y[0] * weight_x[2];
                    _target_value_intp += target_value_intp[1][2] * weight_y[1] * weight_x[2];
                    _target_value_intp += target_value_intp[2][2] * weight_y[2] * weight_x[2];
                    _target_value_intp += target_value_intp[3][2] * weight_y[3] * weight_x[2];
                    _target_value_intp += target_value_intp[0][3] * weight_y[0] * weight_x[3];
                    _target_value_intp += target_value_intp[1][3] * weight_y[1] * weight_x[3];
                    _target_value_intp += target_value_intp[2][3] * weight_y[2] * weight_x[3];
                    _target_value_intp += target_value_intp[3][3] * weight_y[3] * weight_x[3];
                    _mean_target_intp += _target_value_intp;
                    _sum_target_intp += _target_value_intp * _target_value_intp;

                }
            }
        }
 
        _sum_target_intp = _sum_target_intp - _mean_target_intp * _mean_target_intp /float(subset * subset);
        _sum_target_intp = sqrt(_sum_target_intp);


         for (int j = -halfSubset; j <= halfSubset; j++)
        {
            for (int k = -(halfSubset + lane_id); k < (halfSubset + 8 - lane_id); k++)
            {
                int g_index = (g_y + halfWinSize + threadIdx.y + j) * width + g_x + halfWinSize + threadIdx.x + k;
                
                uchar image_value_ref = _origin_image_ref[g_index];
                float mean_image_value_ref = _mean_image[g_index];
                float sigma_image_value_ref = _sigma_image_gpu[g_index];

                float Jacobian[6] = {0};
                Jacobian[0] = _x_grad_image[g_index];
                Jacobian[3] = _y_grad_image[g_index];
                float coord_weight[2] = {0};
                coord_weight[0] = float(j) / float(halfSubset + 1);
                coord_weight[1] = float(k) / float(halfSubset + 1);

                if (k >= -7 && k <= 7)
                {
                    Jacobian[1] = Jacobian[0] * coord_weight[0]; // x;
                    Jacobian[2] = Jacobian[0] * coord_weight[1]; // y;
                    Jacobian[4] = Jacobian[3] * coord_weight[0];
                    Jacobian[5] = Jacobian[3] * coord_weight[1];

                    float invHJacobian[6] = {0};
                    for (int iRow = 0; iRow < 6; iRow++)
                    {
                        for (int iCol = 0; iCol < 6; iCol++)
                        {
                            int g_index_sm = iRow * 6 + iCol;
                            invHJacobian[iCol] += _hessian_inv_image_sm[g_index_sm * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y + thread_index] * Jacobian[iCol];
                        }
                    }



                }
            }
        }
    }
}

__global__ void calNewDisp(int subset, int sideW, int maxIterNum, int width, int height,
                           float *_init_p, uchar *_origin_image_ref, uchar *_origin_image_target,
                           float *_x_grad_image, float *_y_grad_image, float *_mean_image, float *_sigma_image_gpu,
                           float *_hessian_inv_image, float *_origin_disp,
                           float *_opt_disp)
{
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_x = thread_index % BLOCK_DATA_DIM_X;
    int thread_y = thread_index / BLOCK_DATA_DIM_X;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int g_x = blockIdx.x * blockDim.x * NUM_PER_THREAD_X;
    int g_y = blockIdx.y * blockDim.y * NUM_PER_THREAD_Y;
    g_x = (g_x - 2 * blockIdx.x * halfWinSize) < 0 ? 0 : (g_x - 2 * blockIdx.x * halfWinSize);
    g_y = (g_y - 2 * blockIdx.y * halfWinSize) < 0 ? 0 : (g_y - 2 * blockIdx.y * halfWinSize);

    __shared__ float _x_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k
    __shared__ float _y_grad_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k
    __shared__ uchar _src_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y];    // 4k
    __shared__ uchar _target_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y];

    for (int i = 0; i < WARP_SIZE / 2; ++i)
    {
        _x_grad_image_sm[(thread_y * 16 + i) * BLOCK_DATA_DIM_X + thread_x] =
            _x_grad_image[(g_y + thread_y * 16 + i) * width + g_x + thread_x];
        _y_grad_image_sm[(thread_y * 16 + i) * BLOCK_DATA_DIM_X + thread_x] =
            _y_grad_image[(g_y + thread_y * 16 + i) * width + g_x + thread_x];
        _src_image_sm[(thread_y * 16 + i) * BLOCK_DATA_DIM_X + thread_x] =
            _origin_image_ref[(g_y + thread_y * 16 + i) * width + g_x + thread_x];
        _target_image_sm[(thread_y * 16 + i) * BLOCK_DATA_DIM_X + thread_x] =
            _origin_image_target[(g_y + thread_y * 16 + i) * width + g_x + thread_x];
    }

    __syncthreads();

    float hessian[6 * 6] = {0};
    float sum = 0.0f;
    float squa_sum = 0.0f;
    float mean_value = 0.0f;
    float deltaMean = 0.0f;
    float ref_image_value[SUBREGION_NUM] = {0};
    float Jacobian[SUBREGION_NUM][6];
    int n = 0;
    // if ((g_x - halfWinSize) >= 0 && (g_x + halfWinSize) < width && (g_y - halfWinSize) >= 0 && (g_y + halfWinSize) < height)
    {
        for (int j = -halfSubset; j <= halfSubset; j++) // y
        {
            for (int k = -halfSubset; k <= halfSubset; k++) // x
            {
                uchar image_value = _src_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                  halfWinSize + j + threadIdx.x];
                ref_image_value[n] = image_value;

                Jacobian[n][0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + j +
                                                  threadIdx.x];
                Jacobian[n][1] = Jacobian[n][0] * double(k) / double(halfSubset + 1); // x;
                Jacobian[n][2] = Jacobian[n][0] * double(j) / double(halfSubset + 1); // y;
                Jacobian[n][3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + halfWinSize + j +
                                                  threadIdx.x];
                Jacobian[n][4] = Jacobian[n][3] * double(k) / double(halfSubset + 1);
                Jacobian[n][5] = Jacobian[n][3] * double(j) / double(halfSubset + 1);

                n++;
            }
        }
    }

    if (sum == 0.0f)
    {
        return;
    }

    float ref_delta_vec[SUBREGION_NUM] = {0};
    for (int i = 0; i < SUBREGION_NUM; i++)
    {
        ref_delta_vec[i] = ref_image_value[i] - mean_value;
    }

    __syncthreads();

    float invH[6][6] = {0};

    float invHJacob[SUBREGION_NUM * 6] = {0};
    for (int j = 0; j < SUBREGION_NUM; j++)
    {

        invHJacob[j]                     = invH[0][0] * Jacobian[j][0] + invH[0][1] * Jacobian[j][1] + invH[0][2] * Jacobian[j][2] + invH[0][3] * Jacobian[j][3] + invH[0][4] * Jacobian[j][4] + invH[0][5] * Jacobian[j][5];
        invHJacob[j + SUBREGION_NUM]     = invH[1][0] * Jacobian[j][0] + invH[1][1] * Jacobian[j][1] + invH[1][2] * Jacobian[j][2] + invH[1][3] * Jacobian[j][3] + invH[1][4] * Jacobian[j][4] + invH[1][5] * Jacobian[j][5];
        invHJacob[j + 2 * SUBREGION_NUM] = invH[2][0] * Jacobian[j][0] + invH[2][1] * Jacobian[j][1] + invH[2][2] * Jacobian[j][2] + invH[2][3] * Jacobian[j][3] + invH[2][4] * Jacobian[j][4] + invH[2][5] * Jacobian[j][5];
        invHJacob[j + 3 * SUBREGION_NUM] = invH[3][0] * Jacobian[j][0] + invH[3][1] * Jacobian[j][1] + invH[3][2] * Jacobian[j][2] + invH[3][3] * Jacobian[j][3] + invH[3][4] * Jacobian[j][4] + invH[3][5] * Jacobian[j][5];
        invHJacob[j + 4 * SUBREGION_NUM] = invH[4][0] * Jacobian[j][0] + invH[4][1] * Jacobian[j][1] + invH[4][2] * Jacobian[j][2] + invH[4][3] * Jacobian[j][3] + invH[4][4] * Jacobian[j][4] + invH[4][5] * Jacobian[j][5];
        invHJacob[j + 5 * SUBREGION_NUM] = invH[5][0] * Jacobian[j][0] + invH[5][1] * Jacobian[j][1] + invH[5][2] * Jacobian[j][2] + invH[5][3] * Jacobian[j][3] + invH[5][4] * Jacobian[j][4] + invH[5][5] * Jacobian[j][5];
    }

    float warP[3][3] = {{1 + _init_p[1], _init_p[2], _init_p[0]},
                        {_init_p[4], 1 + _init_p[5], _init_p[3]},
                        {0, 0, 1}};
    float thre = 1;
    int Iter = 0;
    float Czncc = 0;
    while (thre > 1e-3 && Iter < maxIterNum || Iter == 0)
    {

        float target_value_intp[SUBREGION_NUM] = {0};
        float target_sum_intp = 0;
        calTargetImageSubRegion(subset, sideW, maxIterNum, &_target_image_sm[0], warP, &target_value_intp[0], &target_sum_intp);
        float target_mean_intp = target_sum_intp / SUBREGION_NUM;
        float target_delta_vec[SUBREGION_NUM] = {0};
        float target_sum_delta = 0.0f;

        for (int j = 0; j < SUBREGION_NUM; j++)
        {
            target_delta_vec[j] = target_value_intp[j] - target_mean_intp;
            target_sum_delta += (target_delta_vec[j] * target_delta_vec[j]);
        }

        float target_delta_sqrt = sqrt(target_sum_delta);

        float deltap[6] = {0};
        for (int j = 0; j < subset * subset; j++)
        {
            float tmp = ref_delta_vec[j] - deltaMean / target_delta_sqrt * target_delta_vec[j];

            deltap[0] += -invHJacob[j] * tmp;
            deltap[1] += -invHJacob[j + SUBREGION_NUM] * tmp;
            deltap[2] += -invHJacob[j + 2 * SUBREGION_NUM] * tmp;
            deltap[3] += -invHJacob[j + 3 * SUBREGION_NUM] * tmp;
            deltap[4] += -invHJacob[j + 4 * SUBREGION_NUM] * tmp;
            deltap[5] += -invHJacob[j + 5 * SUBREGION_NUM] * tmp;
        }
        float M[6] = {1, 1.0f / subset, 1.0f / subset, 1, 1.0f / subset, 1.0f / subset};
        deltap[0] = M[0] * deltap[0];
        deltap[1] = M[1] * deltap[1];
        deltap[2] = M[2] * deltap[2];
        deltap[3] = M[3] * deltap[3];
        deltap[4] = M[4] * deltap[4];
        deltap[5] = M[5] * deltap[5];

        float delta_warp_p[3 * 3] = {1 + deltap[1], deltap[2], deltap[0],
                                     deltap[4], 1 + deltap[5], deltap[3],
                                     0, 0, 1};
        float invwarpdelta[3][3] = {0};
        // getMatrixInverse3(delta_warp_p, 3, invwarpdelta);

        // float warP2[3][3] = {0};
        // warP2[0][0] = warP[0][0] * invwarpdelta[0][0] + warP[0][1] * invwarpdelta[1][0] + warP[0][2] * invwarpdelta[2][0];
        // warP2[0][1] = warP[0][0] * invwarpdelta[0][1] + warP[0][1] * invwarpdelta[1][1] + warP[0][2] * invwarpdelta[2][1];
        // warP2[0][2] = warP[0][0] * invwarpdelta[0][2] + warP[0][1] * invwarpdelta[1][2] + warP[0][2] * invwarpdelta[2][2];
        // warP2[1][0] = warP[1][0] * invwarpdelta[0][0] + warP[1][1] * invwarpdelta[1][0] + warP[1][2] * invwarpdelta[2][0];
        // warP2[1][1] = warP[1][0] * invwarpdelta[0][1] + warP[1][1] * invwarpdelta[1][1] + warP[1][2] * invwarpdelta[2][1];
        // warP2[1][2] = warP[1][0] * invwarpdelta[0][2] + warP[1][1] * invwarpdelta[1][2] + warP[1][2] * invwarpdelta[2][2];
        // warP2[2][0] = warP[2][0] * invwarpdelta[0][0] + warP[2][1] * invwarpdelta[1][0] + warP[2][2] * invwarpdelta[2][0];
        // warP2[2][1] = warP[2][0] * invwarpdelta[0][1] + warP[2][1] * invwarpdelta[1][1] + warP[2][2] * invwarpdelta[2][1];
        // warP2[2][2] = warP[2][0] * invwarpdelta[0][2] + warP[2][1] * invwarpdelta[1][2] + warP[2][2] * invwarpdelta[2][2];

        // warP[0][0] = warP2[0][0]; warP[0][1] = warP2[0][1]; warP[0][2] = warP2[0][2];
        // warP[1][0] = warP2[1][0]; warP[1][1] = warP2[1][1]; warP[1][2] = warP2[1][2];
        // warP[2][0] = warP2[2][0]; warP[2][1] = warP2[2][1]; warP[2][2] = warP2[2][2];

        // float delta_value = deltap[0] * deltap[0] + deltap[3] * deltap[3];
        // thre = sqrt(delta_value);

        // _init_p[0] = warP[0][2];
        // _init_p[1] = warP[0][0] - 1;
        // _init_p[2] = warP[0][1];
        // _init_p[3] = warP[1][2];
        // _init_p[4] = warP[1][0];
        // _init_p[5] = warP[1][1] - 1;
        // float Cznssd = 0;
        // for (int j = 0; j < subset * subset; j++)
        // {
        //     float deltafg = (ref_delta_vec[j] / deltaMean - target_delta_vec[j] / target_delta_sqrt);
        //     Cznssd += deltafg * deltafg;
        // }
        // Czncc = 1 - 0.5 * Cznssd;
        Iter++;
    }
    float delta_disp = _init_p[3];

    _opt_disp[(g_y + halfWinSize + threadIdx.y) * width + g_x + threadIdx.x + halfWinSize] += delta_disp;
}

void CDispOptimizeICGN_GPU::run(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter,
                                cv::Mat &_result)
{

    float MBT_cpu[4][4] = {{-0.166666666666667, 0.5, -0.5, 0.166666666666667},
                       {0.5, -1, 0, 0.666666666666667},
                       {-0.5, 0.5, 0.5, 0.166666666666667},
                       {0.166666666666667, 0, 0, 0}};
    cudaMemcpyToSymbol(MBT, MBT_cpu, sizeof(float) * 16, 0, cudaMemcpyHostToDevice); // 复制数据
    // 生成左图像梯度影像,分为x,y两个方向;
    cv::Mat _x_gradient_image_cpu, _y_gradient_image_cpu;
    _x_gradient_image_cpu.create(_l_image.size(), CV_32FC1);
    _y_gradient_image_cpu.create(_l_image.size(), CV_32FC1);
    // generate_gradient_image(_l_image, _x_gradient_image, _y_gradient_image);
    // // 保存梯度影像;

    float *_x_gradient_image = nullptr;
    float *_y_gradient_image = nullptr;
    generate_gradient_image(_l_image, _x_gradient_image, _y_gradient_image);

    {
        cudaMemcpy(_x_gradient_image_cpu.data, _x_gradient_image, _l_image.rows * _l_image.cols * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(_y_gradient_image_cpu.data, _y_gradient_image, _l_image.rows * _l_image.cols * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cv::imwrite("x_gradient_image_cpu.tif", _x_gradient_image_cpu);
        cv::imwrite("y_gradient_image_cpu.tif", _y_gradient_image_cpu);
    }

    float *hessian_mat = nullptr;
    generate_hessian_mat(subset, sideW, maxIter, _l_image.cols, _l_image.rows, _x_gradient_image, _y_gradient_image,
                         hessian_mat);

    {
        cv::Mat hessian = cv::Mat(36, _l_image.cols * _l_image.rows, CV_32FC1);
        cudaMemcpy(hessian.data, hessian_mat, _l_image.cols * _l_image.rows * 36 * sizeof(float),
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);

        cv::imwrite("./hessian.tif", hessian);
    }

    // 计算均值影像;

    float *_mean_image_gpu = nullptr;
    float *_sigma_image_gpu = nullptr;
    calMeanImage(subset, sideW, _l_image.cols, _l_image.rows, (uchar *)_l_image.data, _mean_image_gpu,
                 _sigma_image_gpu);
    {
        cv::Mat meanImage = cv::Mat(_l_image.rows, _l_image.cols, CV_32FC1, 0.0f);
        cv::Mat sigmaImage = cv::Mat(_l_image.rows, _l_image.cols, CV_32FC1, 0.0f);
        cudaMemcpy((float *)meanImage.data, _mean_image_gpu, sizeof(float) * _l_image.cols * _l_image.rows,
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaMemcpy((float *)sigmaImage.data, _sigma_image_gpu, sizeof(float) * _l_image.cols * _l_image.rows,
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);

        cv::imwrite("./meanImage.tif", meanImage);
        cv::imwrite("./sigmaImage.tif", sigmaImage);
    }
    float *hessian_inv = nullptr;
    calInvHessianImage(subset, sideW, _l_image.cols, _l_image.rows, hessian_mat, hessian_inv);

    {
        cv::Mat hessian_inv_image = cv::Mat(36, _l_image.cols * _l_image.rows, CV_32FC1);
        cudaMemcpy((float *)hessian_inv_image.data, hessian_inv, 36 * sizeof(float) * _l_image.cols * _l_image.rows,
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);

        cv::imwrite("./hessian_inv_image.tif", hessian_inv_image);
    }
    calOptDisp(subset, sideW, maxIter, _l_image.cols, _r_image.rows, _l_image.data, _r_image.data, 
                _x_gradient_image, _y_gradient_image, _mean_image_gpu,_sigma_image_gpu, hessian_inv, 
                (float *)_src_disp.data, (float *)_result.data);

    {
        cv::imwrite("./disp_result.tif", _result);
    }
    printf("3333\n");
    cudaFree(_x_gradient_image);
    _x_gradient_image = nullptr;
    cudaFree(_y_gradient_image);
    _y_gradient_image = nullptr;
    cudaFree(hessian_mat);
    hessian_mat = nullptr;
    printf("4444\n");
}

void CDispOptimizeICGN_GPU::generate_hessian_mat(int subset, int sideW, int maxIter, int width, int height,
                                                 float *_x_gradient_image,
                                                 float *_y_gradient_image, float *&_hessian_mat)
{
    cudaMalloc((void **)&_hessian_mat, width * height * sizeof(float) * 6 * 6);

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
    calHessianMat_kernel_opt_write_back_opt<<<blocks, threads>>>(subset, sideW, width, height, _x_gradient_image,
                                                                 _y_gradient_image, _hessian_mat);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("generate_hessian_mat time: %f ms\n", time);
}

void CDispOptimizeICGN_GPU::generate_gradient_image(cv::Mat &_l_image, float *&_x_gradient_image,
                                                    float *&_y_gradient_image)
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
    generate_gradient_image_kernel<<<blocks, threads>>>(_l_image.cols, _l_image.rows, src_image, _x_gradient_image,
                                                        _y_gradient_image);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("generate_gradient_image_x time: %f ms\n", time);

    cudaFree(src_image);src_image = nullptr;
    return;
}

void CDispOptimizeICGN_GPU::generate_gradient_image(cv::Mat &_l_image, cv::Mat &_x_gradient_image,
                                                    cv::Mat &_y_gradient_image)
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
    generate_gradient_image_kernel<<<blocks, threads>>>(_l_image.cols, _l_image.rows, src_image, _x_dst_image,
                                                        _y_dst_image);
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

void CDispOptimizeICGN_GPU::calOptDisp(int subset, int sideW, int maxIter, int width, int height,
                                       uchar *_origin_image_ref, uchar *_origin_image_target, float *_x_gradient_image,
                                       float *_y_gradient_image, float *_mean_image, float *_sigma_image_gpu,float *_hessian_inv_image,
                                       float *_origin_disp_image, float *_opt_disp_image)
{
    float *_origin_disp_image_gpu = nullptr;
    float *_opt_disp_image_gpu = nullptr;
    uchar *_origin_image_ref_gpu = nullptr;
    uchar *_origin_image_target_gpu = nullptr;
    float p[6] = {0.4686, 0, 0, -0.2116, 0, 0};
    float *_init_p_gpu = nullptr;
    cudaMalloc((void **)&_origin_disp_image_gpu, width * height * sizeof(float));
    cudaMalloc((void **)&_opt_disp_image_gpu, width * height * sizeof(float));
    cudaMalloc((void **)&_origin_image_ref_gpu, width * height * sizeof(uchar));
    cudaMalloc((void **)&_origin_image_target_gpu, width * height * sizeof(uchar));
    cudaMalloc((void **)&_init_p_gpu, 6 * sizeof(float));

    cudaMemcpy(_origin_disp_image_gpu, _origin_disp_image, width * height * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_origin_image_ref_gpu, _origin_image_ref, width * height * sizeof(uchar),
               cudaMemcpyHostToDevice);
    cudaMemcpy(_origin_image_target_gpu, _origin_image_target, width * height * sizeof(uchar),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_opt_disp_image_gpu, _opt_disp_image, width * height * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_init_p_gpu, &p[0], 6 * sizeof(float), cudaMemcpyHostToDevice);

    //cudaMemset(_origin_image_ref_gpu, 0, width * height * sizeof(uchar));
    // cudaMemset(_y_gradient_image, 0, width * height * sizeof(float));

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;

    dim3 threads(16, 16);
    dim3 blocks((width - 2 * halfWinSize + BLOCK_THREAD_DIM_X - 1) / (BLOCK_THREAD_DIM_X),
                (height - 2 * halfWinSize + BLOCK_THREAD_DIM_X - 1) / (BLOCK_THREAD_DIM_X));

    printf("width: %d, height: %d, blocks.x: %d, blocks.y: %d, threads.x: %d, threads.y: %d\n",
           width, height, blocks.x, blocks.y, threads.x, threads.y);

    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    calOptDisp_singleThreadBlock<<<blocks, threads>>>(subset, sideW, maxIter, width, height, _init_p_gpu, _origin_image_ref_gpu,
                                    _origin_image_target_gpu,
                                    _x_gradient_image, _y_gradient_image,_mean_image, _sigma_image_gpu,
                                    _hessian_inv_image, _origin_disp_image_gpu, _opt_disp_image_gpu);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("calOptDisp time: %f ms\n", time);

    cudaMemcpy(_opt_disp_image, _opt_disp_image_gpu, width * height * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(_origin_disp_image_gpu);
    _origin_disp_image_gpu = nullptr;
    cudaFree(_opt_disp_image_gpu);
    _opt_disp_image_gpu = nullptr;
    cudaFree(_origin_image_ref_gpu);
    _origin_image_ref_gpu = nullptr;
    cudaFree(_origin_image_target_gpu);
    _origin_image_target_gpu = nullptr;
    cudaFree(_init_p_gpu);
    _init_p_gpu = nullptr;
}

void CDispOptimizeICGN_GPU::calMeanImage(int subset, int sideW, int width, int height, uchar *_origin_image_ref,
                                         float *&_mean_image_gpu, float *&_sigma_image_gpu)
{
    uchar *_src_image_gpu = nullptr;
    cudaMalloc((void **)&_src_image_gpu, sizeof(uchar) * width * height);
    cudaMalloc((void **)&_mean_image_gpu, sizeof(float) * width * height);
    cudaMalloc((void **)&_sigma_image_gpu, sizeof(float) * width * height);

    cudaMemcpy(_src_image_gpu, (uchar *)_origin_image_ref, sizeof(uchar) * width * height,
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_THREAD_DIM_X, BLOCK_THREAD_DIM_Y);
    dim3 blocks((width - 2 * 12 + threads.x - 1) / (threads.x),
                (height - 2 * 12 + threads.y - 1) / (threads.y));
    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    calMeanAndSigma<<<blocks, threads>>>(subset, sideW, width, height,
                                         _src_image_gpu, _mean_image_gpu, _sigma_image_gpu);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("generate_mean_image time: %f ms\n", time);

    cudaFree(_src_image_gpu);_src_image_gpu = nullptr;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CDispOptimizeICGN_GPU::calInvHessianImage(int subset, int sideW, int width, int height, float *_hessian_mat,
                                               float *&_hessian_mat_inv)
{
    cudaMalloc((void **)&_hessian_mat_inv, sizeof(float) * width * height * 36);
    dim3 threads(BLOCK_THREAD_DIM_X, BLOCK_THREAD_DIM_Y);
    dim3 blocks((width - 2 * 12 + threads.x - 1) / (threads.x),
                (height - 2 * 12 + threads.y - 1) / (threads.y));
    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    calInvHessian<<<blocks, threads>>>(subset, sideW, width, height,
                                       _hessian_mat, _hessian_mat_inv);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("generate_InvHessianImage time: %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
