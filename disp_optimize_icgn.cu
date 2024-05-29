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
#define WARP_SIZE 32
#define SUBSET_SIZE 15
#define SUBREGION_NUM (SUBSET_SIZE*SUBSET_SIZE)
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
                Jacobian[0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + j) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + k + threadIdx.x];
                Jacobian[1] = Jacobian[0] * double(j) / double(halfSubset + 1); // x;
                Jacobian[2] = Jacobian[0] * double(k) / double(halfSubset + 1); // y;
                Jacobian[3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + j) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + halfWinSize + k + threadIdx.x];
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

__global__ void calHessianMat_kernel_opt(int subset, int sideW, int width, int height, float *_x_grad_image, float *_y_grad_image,
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
                Jacobian[0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + j + threadIdx.x];
                Jacobian[1] = Jacobian[0] * double(k) / double(halfSubset + 1); // x;
                Jacobian[2] = Jacobian[0] * double(j) / double(halfSubset + 1); // y;
                Jacobian[3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + halfWinSize + j + threadIdx.x];
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
        _hessian_mat[(g_y + halfWinSize + threadIdx.y) * width + g_x + threadIdx.x + halfWinSize + i * width * height] = hessian[i];
        //_hessian_mat[i] = hessian[i];
    }
}

__global__ void calHessianMat_kernel_opt_write_back(int subset, int sideW, int width, int height, float *_x_grad_image, float *_y_grad_image,
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
                Jacobian[0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + j + threadIdx.x];
                Jacobian[1] = Jacobian[0] * double(k) / double(halfSubset + 1); // x;
                Jacobian[2] = Jacobian[0] * double(j) / double(halfSubset + 1); // y;
                Jacobian[3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + halfWinSize + j + threadIdx.x];
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
    int block_index = blockIdx.y * gridDim.x + blockIdx.x;
    
    for (int i = 0; i < 6 * 6; i++)
    {
        int g_index = (halfWinSize) * width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y + thread_index + i * width * height;
        // int block_start_index = (halfWinSize) * width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y;
        // if(blockIdx.y == 0 && blockIdx.x == 0 && i == 0|| blockIdx.y == 0 && blockIdx.x == 1 && i == 0)
        // {
        //     printf("i: %d, blockIdx.y: %d, blockIdx.x: %d, threadIdx.y: %d, threadIdx.x: %d, thread_index: %d, g_index: %d,block_start_index: %d\n", 
        //     i, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x,thread_index, g_index, block_start_index);
        //
        _hessian_mat[g_index] = hessian[i];
    }

}

__device__ float getA(float *hessian, int n)
{
    float ans = 0;
    int iter_num = 1;
    while (iter_num <= n)
    {
        if(iter_num == 1){
            ans = hessian[0];
            iter_num++;
            continue;
        }
        float temp[6 * 6] = {0.0};
        int i, j, k;
        for (i = 0; i < iter_num; i++)
        {
            for (j = 0; j < iter_num - 1; j++)
            {
                for (k = 0; k < iter_num - 1; k++)
                {
                    temp[j * 6 + k] = hessian[(j + 1) * 6 + ((k >= i) ? k + 1 : k)];
                }
            }
            if (i % 2 == 0)
            {
                ans += hessian[i] * ans;
            }
            else
            {
                ans -= hessian[i] * ans;
            }
        }

        iter_num++;
    }
    return ans;
}

__device__ void getAStart(float *hessian, int n, float (*ans)[6])
{
    if (n == 1)
    {
        ans[0][0] = 1;
    }
    int i, j, k, t;
    float temp[6 * 6];
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n - 1; k++)
            {
                for (t = 0; t < n - 1; t++)
                {
                    temp[k * 6 + t] = hessian[(k >= i ? k + 1 : k) * 6 + (t >= j ? t + 1 : t)];
                }
            }

            ans[j][i] = getA(temp, n - 1);
            if ((i + j) % 2 == 1)
            {
                ans[j][i] = -ans[j][i];
            }
        }
    }
}

__device__ void getMatrixInverse(float *hessian, int n, float (*des)[6])
{
    float flag = getA(hessian, n);
    float t[6][6];
    if (flag == 0)
    {
        return;
    }
    else
    {
        getAStart(hessian, n, t);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                des[i][j] = t[i][j] / flag;
            }
        }
    }
}

// 按第一行展开计算|A|
__device__ float getA3(float *arcs, int n)
{
    float ans = 0;
    int iter_num = 1;
    while (iter_num <= n)
    {
        if(iter_num == 1){
            ans = arcs[0];
            iter_num++;
            continue;
        }
        float temp[3 * 3] = {0.0};
        int i, j, k;
        for (i = 0; i < iter_num; i++)
        {
            for (j = 0; j < iter_num - 1; j++)
            {
                for (k = 0; k < iter_num - 1; k++)
                {
                    temp[j * 3 + k] = arcs[(j + 1) * 3 + ((k >= i) ? k + 1 : k)];
                }
            }
            if (i % 2 == 0)
            {
                ans += arcs[i] * ans;
            }
            else
            {
                ans -= arcs[i] * ans;
            }
        }
        iter_num++;
    }

    return ans;
}


// 计算每一行每一列的每个元素所对应的余子式，组成A*
__device__ int getAStart3(float *arcs, int n, float ans[3][3])
{
    if (n == 1)
    {
        ans[0][0] = 1;
        return 0;
    }
    int i, j, k, t;
    float temp[3*3];
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n - 1; k++)
            {
                for (t = 0; t < n - 1; t++)
                {
                    temp[k* 3 + t] = arcs[(k >= i ? k + 1 : k) * 3 + (t >= j ? t + 1 : t)];
                }
            }

            ans[j][i] = getA3(temp, n - 1);
            if ((i + j) % 2 == 1)
            {
                ans[j][i] = -ans[j][i];
            }
        }
    }
    return 1;
}


__device__ void getMatrixInverse3(float *src, int n, float (*des)[3])
{
    float flag = getA3(src, n);
    float t[3][3];
    if (flag == 0)
    {
        return;
    }
    else
    {
        getAStart3(src, n, t);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                des[i][j] = t[i][j] / flag;
            }
        }
    }
    return;
}

__global__ void calMeanAndSigma(int subset, int sideW, int width, int height, uchar *_origin_src_image,
                                float *_delta_image, float *_sigma_image)
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
    __shared__ uchar _src_image_sm[BLOCK_DATA_DIM_X * BLOCK_DATA_DIM_Y]; // 4k

    for (int i = 0; i < WARP_SIZE / 2; i++)
    {
        _src_image_sm[(thread_y * 16 + i) * BLOCK_DATA_DIM_X + thread_x] =
            _origin_src_image[(g_y + thread_y * 16 + i) * width + g_x + thread_x];
    }
    __syncthreads();
    float sum = 0.0f;
    for (int j = -halfSubset; j <= halfSubset; j++) // y
    {
        for (int k = -halfSubset; k <= halfSubset; k++) // x
        {
            uchar value = _src_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + j + threadIdx.x];
            sum += value;
        }
    }
    float mean_value = float(sum) / float(subset * subset);
    __syncthreads();
}

__device__ void calTargetImageSubRegion(int subset, int sideW, int maxIterNum, uchar *_origin_image_target,
                                        float (*warP)[3], float *_target_value_intp, float *_sum_target_intp)
{

    float MBT[4][4] = {{-0.166666666666667, 0.5, -0.5, 0.166666666666667},
                       {0.5, -1, 0, 0.666666666666667},
                       {-0.5, 0.5, 0.5, 0.166666666666667},
                       {0.166666666666667, 0, 0, 0}};
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
__global__ void calNewDisp(int subset, int sideW, int maxIterNum, int width, int height,
                           float *_init_p, uchar *_origin_image_ref, uchar *_origin_image_target,
                           float *_x_grad_image, float *_y_grad_image, float *_origin_disp,
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
    //if ((g_x - halfWinSize) >= 0 && (g_x + halfWinSize) < width && (g_y - halfWinSize) >= 0 && (g_y + halfWinSize) < height)
    {
        for (int j = -halfSubset; j <= halfSubset; j++) // y
        {
            for (int k = -halfSubset; k <= halfSubset; k++) // x
            {
                uchar image_value = _src_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y +
                                                  halfWinSize + j + threadIdx.x];
                sum += image_value;
                squa_sum += image_value * image_value;
                ref_image_value[n] = image_value;
                
                Jacobian[n][0] = _x_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + j + threadIdx.x];
                Jacobian[n][1] = Jacobian[n][0] * double(k) / double(halfSubset + 1); // x;
                Jacobian[n][2] = Jacobian[n][0] * double(j) / double(halfSubset + 1); // y;
                Jacobian[n][3] = _y_grad_image_sm[(halfWinSize + threadIdx.y + k) * BLOCK_THREAD_DIM_X * NUM_PER_THREAD_X + halfWinSize + j + threadIdx.x];
                Jacobian[n][4] = Jacobian[n][3] * double(k) / double(halfSubset + 1);
                Jacobian[n][5] = Jacobian[n][3] * double(j) / double(halfSubset + 1);

                if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0)
                {
                    printf("Jacobian[%d][0]: %lf,Jacobian[%d][1]: %lf,Jacobian[%d][2]: %lf,Jacobian[%d][3]: %lf,Jacobian[%d][4]: %lf,Jacobian[%d][5]: %lf\n", 
                            n, Jacobian[n][0], n, Jacobian[n][1],n, Jacobian[n][2], 
                            n, Jacobian[n][3], n, Jacobian[n][4],n, Jacobian[n][5]);
                }
                hessian[0]  += Jacobian[n][0] * Jacobian[n][0];
                hessian[1]  += Jacobian[n][0] * Jacobian[n][1];
                hessian[2]  += Jacobian[n][0] * Jacobian[n][2];
                hessian[3]  += Jacobian[n][0] * Jacobian[n][3];
                hessian[4]  += Jacobian[n][0] * Jacobian[n][4];
                hessian[5]  += Jacobian[n][0] * Jacobian[n][5];
                hessian[6]  += Jacobian[n][1] * Jacobian[n][0];
                hessian[7]  += Jacobian[n][1] * Jacobian[n][1];
                hessian[8]  += Jacobian[n][1] * Jacobian[n][2];
                hessian[9]  += Jacobian[n][1] * Jacobian[n][3];
                hessian[10] += Jacobian[n][1] * Jacobian[n][4];
                hessian[11] += Jacobian[n][1] * Jacobian[n][5];
                hessian[12] += Jacobian[n][2] * Jacobian[n][0];
                hessian[13] += Jacobian[n][2] * Jacobian[n][1];
                hessian[14] += Jacobian[n][2] * Jacobian[n][2];
                hessian[15] += Jacobian[n][2] * Jacobian[n][3];
                hessian[16] += Jacobian[n][2] * Jacobian[n][4];
                hessian[17] += Jacobian[n][2] * Jacobian[n][5];
                hessian[18] += Jacobian[n][3] * Jacobian[n][0];
                hessian[19] += Jacobian[n][3] * Jacobian[n][1];
                hessian[20] += Jacobian[n][3] * Jacobian[n][2];
                hessian[21] += Jacobian[n][3] * Jacobian[n][3];
                hessian[22] += Jacobian[n][3] * Jacobian[n][4];
                hessian[23] += Jacobian[n][3] * Jacobian[n][5];
                hessian[24] += Jacobian[n][4] * Jacobian[n][0];
                hessian[25] += Jacobian[n][4] * Jacobian[n][1];
                hessian[26] += Jacobian[n][4] * Jacobian[n][2];
                hessian[27] += Jacobian[n][4] * Jacobian[n][3];
                hessian[28] += Jacobian[n][4] * Jacobian[n][4];
                hessian[29] += Jacobian[n][4] * Jacobian[n][5];
                hessian[30] += Jacobian[n][5] * Jacobian[n][0];
                hessian[31] += Jacobian[n][5] * Jacobian[n][1];
                hessian[32] += Jacobian[n][5] * Jacobian[n][2];
                hessian[33] += Jacobian[n][5] * Jacobian[n][3];
                hessian[34] += Jacobian[n][5] * Jacobian[n][4];
                hessian[35] += Jacobian[n][5] * Jacobian[n][5];
                if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0)
                {
                    printf("000 n: %d, hessian[35]: %lf\n",n, hessian[35]);
                }
                n++;
            }
        }
    }
    
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0 && threadIdx.x == 0)
    // {
    //     printf("1111\n");
    // }
    // if (sum == 0.0f)
    // {
    //     return;
    // }
    // mean_value = float(sum) / float(SUBREGION_NUM);
    // deltaMean = squa_sum - sum * mean_value;
    // float ref_delta_vec[SUBREGION_NUM] = {0};
    // for (int i = 0; i < SUBREGION_NUM; i++)
    // {
    //     ref_delta_vec[i] = ref_image_value[i] - mean_value;
    //     // if (blockIdx.x == 40 && blockIdx.y == 40 && threadIdx.y == 0 && threadIdx.x == 0)
    //     // {
    //     //     printf("sum: %lf, mean_value: %lf, ref_delta_vec[i]: %lf, deltaMean: %lf, ref_image_value[i]: %lf\n",
    //     //            sum, mean_value, ref_delta_vec[i], deltaMean, ref_image_value[i]);
    //     // }
    // }

    __syncthreads();
    if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0)
    {
        printf("111 n: %d, hessian[35]: %lf\n", n, hessian[35]);
    }
    // if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0)
    // {
    //     printf("111 hessian[0]: %lf,hessian[1]: %lf,hessian[2]: %lf,hessian[3]: %lf,hessian[4]: %lf,hessian[5]: %lf,\
    //             hessian[6]: %lf,hessian[7]: %lf,hessian[8]: %lf,hessian[9]: %lf,hessian[10]: %lf,hessian[11]: %lf,\
    //             hessian[12]: %lf,hessian[13]: %lf,hessian[14]: %lf,hessian[15]: %lf,hessian[16]: %lf,hessian[17]: %lf,\
    //             hessian[18]: %lf,hessian[19]: %lf,hessian[20]: %lf,hessian[21]: %lf,hessian[22]: %lf,hessian[23]: %lf,\
    //             hessian[24]: %lf,hessian[25]: %lf,hessian[26]: %lf,hessian[27]: %lf,hessian[28]: %lf,hessian[29]: %lf,\
    //             hessian[30]: %lf,hessian[31]: %lf,hessian[32]: %lf,hessian[33]: %lf,hessian[34]: %lf,hessian[35]: %lf\n",
    //             hessian[0] ,hessian[1] ,hessian[2] ,hessian[3] ,hessian[4] ,hessian[5] ,
    //             hessian[6] ,hessian[7] ,hessian[8] ,hessian[9] ,hessian[10],hessian[11],
    //             hessian[12],hessian[13],hessian[14],hessian[15],hessian[16],hessian[17],
    //             hessian[18],hessian[19],hessian[20],hessian[21],hessian[22],hessian[23],
    //             hessian[24],hessian[25],hessian[26],hessian[27],hessian[28],hessian[29],
    //             hessian[30],hessian[31],hessian[32],hessian[33],hessian[34],hessian[35]);
    //     // printf("invH[0][0]: %lf, invH[0][1]: %lf,invH[0][2]: %lf,invH[0][3]: %lf,invH[0][4]: %lf,invH[0][5]: %lf,\
    //     //         invH[1][0]: %lf, invH[1][1]: %lf,invH[1][2]: %lf,invH[1][3]: %lf,invH[1][4]: %lf,invH[1][5]: %lf,\
    //     //         invH[2][0]: %lf, invH[2][1]: %lf,invH[2][2]: %lf,invH[2][3]: %lf,invH[2][4]: %lf,invH[2][5]: %lf,\
    //     //         invH[3][0]: %lf, invH[3][1]: %lf,invH[3][2]: %lf,invH[3][3]: %lf,invH[3][4]: %lf,invH[3][5]: %lf,\
    //     //         invH[4][0]: %lf, invH[4][1]: %lf,invH[4][2]: %lf,invH[4][3]: %lf,invH[4][4]: %lf,invH[4][5]: %lf,\
    //     //         invH[5][0]: %lf, invH[5][1]: %lf,invH[5][2]: %lf,invH[5][3]: %lf,invH[5][4]: %lf,invH[5][5]: %lf\n",
    //     //         invH[0][0], invH[0][1],invH[0][2],invH[0][3],invH[0][4],invH[0][5],
    //     //         invH[1][0], invH[1][1],invH[1][2],invH[1][3],invH[1][4],invH[1][5],
    //     //         invH[2][0], invH[2][1],invH[2][2],invH[2][3],invH[2][4],invH[2][5],
    //     //         invH[3][0], invH[3][1],invH[3][2],invH[3][3],invH[3][4],invH[3][5],
    //     //         invH[4][0], invH[4][1],invH[4][2],invH[4][3],invH[4][4],invH[4][5],
    //     //         invH[5][0], invH[5][1],invH[5][2],invH[5][3],invH[5][4],invH[5][5]);
    // }
    float invH[6][6] = {0};
    getMatrixInverse(hessian, 6, invH);
    if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0)
    {
        printf("222 n: %d, hessian[35]: %lf\n", n, hessian[35]);
    }
    if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < 36; i++)
        {
            printf("hessian[%d]: %f\n",i,hessian[i]);
        }

        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                printf("invH[%d][%d]: %f\n", i ,j, invH[i][j]);
            }
            
        }
        
        
        // printf("333 hessian[0]: %f,hessian[1]: %f,hessian[2]: %f,hessian[3]: %f,hessian[4]: %f,hessian[5]: %f,\
        //         hessian[6]: %f,hessian[7]: %f,hessian[8]: %f,hessian[9]: %f,hessian[10]: %f,hessian[11]: %f,\
        //         hessian[12]: %f,hessian[13]: %f,hessian[14]: %f,hessian[15]: %f,hessian[16]: %f,hessian[17]: %f,\
        //         hessian[18]: %f,hessian[19]: %f,hessian[20]: %f,hessian[21]: %f,hessian[22]: %f,hessian[23]: %f,\
        //         hessian[24]: %f,hessian[25]: %f,hessian[26]: %f,hessian[27]: %f,hessian[28]: %f,hessian[29]: %f,\
        //         hessian[30]: %f,hessian[31]: %f,hessian[32]: %f,hessian[33]: %f,hessian[34]: %f,hessian[35]: %f\n",
        //         hessian[0] ,hessian[1] ,hessian[2] ,hessian[3] ,hessian[4] ,hessian[5] ,
        //         hessian[6] ,hessian[7] ,hessian[8] ,hessian[9] ,hessian[10],hessian[11],
        //         hessian[12],hessian[13],hessian[14],hessian[15],hessian[16],hessian[17],
        //         hessian[18],hessian[19],hessian[20],hessian[21],hessian[22],hessian[23],
        //         hessian[24],hessian[25],hessian[26],hessian[27],hessian[28],hessian[29],
        //         hessian[30],hessian[31],hessian[32],hessian[33],hessian[34],hessian[35]);
        // printf("invH[0][0]: %lf, invH[0][1]: %lf,invH[0][2]: %lf,invH[0][3]: %lf,invH[0][4]: %lf,invH[0][5]: %lf,\
        //         invH[1][0]: %lf, invH[1][1]: %lf,invH[1][2]: %lf,invH[1][3]: %lf,invH[1][4]: %lf,invH[1][5]: %lf,\
        //         invH[2][0]: %lf, invH[2][1]: %lf,invH[2][2]: %lf,invH[2][3]: %lf,invH[2][4]: %lf,invH[2][5]: %lf,\
        //         invH[3][0]: %lf, invH[3][1]: %lf,invH[3][2]: %lf,invH[3][3]: %lf,invH[3][4]: %lf,invH[3][5]: %lf,\
        //         invH[4][0]: %lf, invH[4][1]: %lf,invH[4][2]: %lf,invH[4][3]: %lf,invH[4][4]: %lf,invH[4][5]: %lf,\
        //         invH[5][0]: %lf, invH[5][1]: %lf,invH[5][2]: %lf,invH[5][3]: %lf,invH[5][4]: %lf,invH[5][5]: %lf\n",
        //         invH[0][0], invH[0][1],invH[0][2],invH[0][3],invH[0][4],invH[0][5],
        //         invH[1][0], invH[1][1],invH[1][2],invH[1][3],invH[1][4],invH[1][5],
        //         invH[2][0], invH[2][1],invH[2][2],invH[2][3],invH[2][4],invH[2][5],
        //         invH[3][0], invH[3][1],invH[3][2],invH[3][3],invH[3][4],invH[3][5],
        //         invH[4][0], invH[4][1],invH[4][2],invH[4][3],invH[4][4],invH[4][5],
        //         invH[5][0], invH[5][1],invH[5][2],invH[5][3],invH[5][4],invH[5][5]);
    }
    // float invHJacob[SUBREGION_NUM * 6] = {0};
    // for (int j = 0; j < SUBREGION_NUM; j++)
    // {
        
    //     invHJacob[j]                     = invH[0][0] * Jacobian[j][0] + invH[0][1] * Jacobian[j][1] + invH[0][2] * Jacobian[j][2] + invH[0][3] * Jacobian[j][3] + invH[0][4] * Jacobian[j][4] + invH[0][5] * Jacobian[j][5];
    //     invHJacob[j + SUBREGION_NUM]     = invH[1][0] * Jacobian[j][0] + invH[1][1] * Jacobian[j][1] + invH[1][2] * Jacobian[j][2] + invH[1][3] * Jacobian[j][3] + invH[1][4] * Jacobian[j][4] + invH[1][5] * Jacobian[j][5];
    //     invHJacob[j + 2 * SUBREGION_NUM] = invH[2][0] * Jacobian[j][0] + invH[2][1] * Jacobian[j][1] + invH[2][2] * Jacobian[j][2] + invH[2][3] * Jacobian[j][3] + invH[2][4] * Jacobian[j][4] + invH[2][5] * Jacobian[j][5];
    //     invHJacob[j + 3 * SUBREGION_NUM] = invH[3][0] * Jacobian[j][0] + invH[3][1] * Jacobian[j][1] + invH[3][2] * Jacobian[j][2] + invH[3][3] * Jacobian[j][3] + invH[3][4] * Jacobian[j][4] + invH[3][5] * Jacobian[j][5];
    //     invHJacob[j + 4 * SUBREGION_NUM] = invH[4][0] * Jacobian[j][0] + invH[4][1] * Jacobian[j][1] + invH[4][2] * Jacobian[j][2] + invH[4][3] * Jacobian[j][3] + invH[4][4] * Jacobian[j][4] + invH[4][5] * Jacobian[j][5];
    //     invHJacob[j + 5 * SUBREGION_NUM] = invH[5][0] * Jacobian[j][0] + invH[5][1] * Jacobian[j][1] + invH[5][2] * Jacobian[j][2] + invH[5][3] * Jacobian[j][3] + invH[5][4] * Jacobian[j][4] + invH[5][5] * Jacobian[j][5];
    // }

    // float warP[3][3] = {{1 + _init_p[1], _init_p[2], _init_p[0]},
    //                     {_init_p[4], 1 + _init_p[5], _init_p[3]},
    //                     {0, 0, 1}};
    // float thre = 1;
    // int Iter = 0;
    // float Czncc = 0;
    // while (thre > 1e-3 && Iter < maxIterNum || Iter == 0)
    // {
    //     // if(blockIdx.x == 38 && blockIdx.y == 0 && threadIdx.y == 0 && threadIdx.x == 0){
    //     //     printf("Iter: %d,thre: %f\n", Iter, thre);
    //     // }
        
    //     float target_value_intp[SUBREGION_NUM] = {0};
    //     float target_sum_intp = 0;
    //     calTargetImageSubRegion(subset, sideW, maxIterNum, &_target_image_sm[0], warP, &target_value_intp[0], &target_sum_intp);
    //     float target_mean_intp = target_sum_intp / SUBREGION_NUM;
    //     float target_delta_vec[SUBREGION_NUM] = {0};
    //     float target_sum_delta = 0.0f;
        
    //     for (int j = 0; j < SUBREGION_NUM; j++)
    //     {
    //         target_delta_vec[j] = target_value_intp[j] - target_mean_intp;
    //         target_sum_delta += (target_delta_vec[j] * target_delta_vec[j]);
    //     }
        
    //     float target_delta_sqrt = sqrt(target_sum_delta);
        
    //     float deltap[6] = {0};
    //     for (int j = 0; j < subset * subset; j++)
    //     {
    //         float tmp = ref_delta_vec[j] - deltaMean / target_delta_sqrt * target_delta_vec[j];
    //         // if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0)
    //         // {
    //         //     printf("j: %d,tmp: %f, invHJacob[%d]: %lf,ref_delta_vec[j]: %lf, deltaMean: %lf,target_delta_sqrt: %lf, target_delta_vec[j]: %lf\n",
    //         //             j, tmp, j, invHJacob[j],ref_delta_vec[j], deltaMean,target_delta_sqrt, target_delta_vec[j]);
    //         // }
    //         deltap[0] += -invHJacob[j] * tmp;
    //         deltap[1] += -invHJacob[j + SUBREGION_NUM] * tmp;
    //         deltap[2] += -invHJacob[j + 2 * SUBREGION_NUM] * tmp;
    //         deltap[3] += -invHJacob[j + 3 * SUBREGION_NUM] * tmp;
    //         deltap[4] += -invHJacob[j + 4 * SUBREGION_NUM] * tmp;
    //         deltap[5] += -invHJacob[j + 5 * SUBREGION_NUM] * tmp;
    //     }
    //     float M[6] = {1, 1.0f / subset, 1.0f / subset, 1, 1.0f / subset, 1.0f / subset};
    //     deltap[0] = M[0] * deltap[0];
    //     deltap[1] = M[1] * deltap[1];
    //     deltap[2] = M[2] * deltap[2];
    //     deltap[3] = M[3] * deltap[3];
    //     deltap[4] = M[4] * deltap[4];
    //     deltap[5] = M[5] * deltap[5];
    //     // if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0)
    //     // {
    //     //     printf("deltap[0]: %lf, deltap[1]: %lf, deltap[2]: %lf, deltap[3]: %lf,deltap[4]: %lf, deltap[5]: %lf\n",
    //     //            deltap[0], deltap[1], deltap[2], deltap[3], deltap[4], deltap[5]);
    //     // }
    //     float delta_warp_p[3 * 3] = {1 + deltap[1], deltap[2], deltap[0],
    //                                   deltap[4], 1 + deltap[5], deltap[3],
    //                                   0, 0, 1};
    //     float invwarpdelta[3][3] = {0};
    //     getMatrixInverse3(delta_warp_p, 3, invwarpdelta);


    //     // float warP2[3][3] = {0};
    //     // warP2[0][0] = warP[0][0] * invwarpdelta[0][0] + warP[0][1] * invwarpdelta[1][0] + warP[0][2] * invwarpdelta[2][0];
    //     // warP2[0][1] = warP[0][0] * invwarpdelta[0][1] + warP[0][1] * invwarpdelta[1][1] + warP[0][2] * invwarpdelta[2][1];
    //     // warP2[0][2] = warP[0][0] * invwarpdelta[0][2] + warP[0][1] * invwarpdelta[1][2] + warP[0][2] * invwarpdelta[2][2];
    //     // warP2[1][0] = warP[1][0] * invwarpdelta[0][0] + warP[1][1] * invwarpdelta[1][0] + warP[1][2] * invwarpdelta[2][0];
    //     // warP2[1][1] = warP[1][0] * invwarpdelta[0][1] + warP[1][1] * invwarpdelta[1][1] + warP[1][2] * invwarpdelta[2][1];
    //     // warP2[1][2] = warP[1][0] * invwarpdelta[0][2] + warP[1][1] * invwarpdelta[1][2] + warP[1][2] * invwarpdelta[2][2];
    //     // warP2[2][0] = warP[2][0] * invwarpdelta[0][0] + warP[2][1] * invwarpdelta[1][0] + warP[2][2] * invwarpdelta[2][0];
    //     // warP2[2][1] = warP[2][0] * invwarpdelta[0][1] + warP[2][1] * invwarpdelta[1][1] + warP[2][2] * invwarpdelta[2][1];
    //     // warP2[2][2] = warP[2][0] * invwarpdelta[0][2] + warP[2][1] * invwarpdelta[1][2] + warP[2][2] * invwarpdelta[2][2];

    //     // warP[0][0] = warP2[0][0]; warP[0][1] = warP2[0][1]; warP[0][2] = warP2[0][2];
    //     // warP[1][0] = warP2[1][0]; warP[1][1] = warP2[1][1]; warP[1][2] = warP2[1][2];
    //     // warP[2][0] = warP2[2][0]; warP[2][1] = warP2[2][1]; warP[2][2] = warP2[2][2];

    //     // float delta_value = deltap[0] * deltap[0] + deltap[3] * deltap[3];
    //     // thre = sqrt(delta_value);
        
    //     // _init_p[0] = warP[0][2];
    //     // _init_p[1] = warP[0][0] - 1;
    //     // _init_p[2] = warP[0][1];
    //     // _init_p[3] = warP[1][2];
    //     // _init_p[4] = warP[1][0];
    //     // _init_p[5] = warP[1][1] - 1;
    //     // float Cznssd = 0;
    //     // for (int j = 0; j < subset * subset; j++)
    //     // {
    //     //     float deltafg = (ref_delta_vec[j] / deltaMean - target_delta_vec[j] / target_delta_sqrt);
    //     //     Cznssd += deltafg * deltafg;
    //     // }
    //     // Czncc = 1 - 0.5 * Cznssd;
    //     Iter++;
    // }
    // float delta_disp = _init_p[3];
    // // if (blockIdx.x == 39 && blockIdx.y == 2 && threadIdx.y == 0 && threadIdx.x == 0){
    // //     printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.y: %d,threadIdx.x: %d, delta_disp: %f\n", 
    // //         blockIdx.x, blockIdx.y, threadIdx.y, threadIdx.x, delta_disp);
    // // }
    
    // _opt_disp[(g_y + halfWinSize + threadIdx.y) * width + g_x + threadIdx.x + halfWinSize] += delta_disp;
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

    double *hessian_mat = nullptr;
    generate_hessian_mat(subset, sideW, maxIter, _l_image.cols, _l_image.rows, _x_gradient_image, _y_gradient_image, hessian_mat);

    cv::Mat hessian = cv::Mat(36, _l_image.cols * _l_image.rows, CV_64FC1);
    cudaMemcpy(hessian.data, hessian_mat, _l_image.cols * _l_image.rows * 36 * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cv::imwrite("./hessian.tif", hessian);
    // calOptDisp(subset, sideW, maxIter, _l_image.cols, _r_image.rows, _l_image.data, _r_image.data, _x_gradient_image,
    //            _y_gradient_image, (float *)_src_disp.data, (float *)_result.data);
    
    cudaFree(_x_gradient_image);_x_gradient_image = nullptr;
    cudaFree(_y_gradient_image);_y_gradient_image = nullptr;
    cudaFree(hessian_mat);hessian_mat = nullptr;
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
    calHessianMat_kernel_opt_write_back<<<blocks, threads>>>(subset, sideW, width, height, _x_gradient_image, _y_gradient_image, _hessian_mat);
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

void CDispOptimizeICGN_GPU::calOptDisp(int subset, int sideW, int maxIter, int width, int height,
                                       uchar *_origin_image_ref, uchar *_origin_image_target, float *_x_gradient_image,
                                       float *_y_gradient_image, float *_origin_disp_image, float *_opt_disp_image)
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
    cudaMalloc((void**)&_init_p_gpu, 6 * sizeof(float));
    cudaMemcpy(_origin_disp_image_gpu, _origin_disp_image, width * height * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(_origin_image_ref_gpu, _origin_image_ref, width * height * sizeof(uchar),
               cudaMemcpyHostToDevice);
    cudaMemcpy(_origin_image_target_gpu, _origin_image_target, width * height * sizeof(uchar),
               cudaMemcpyHostToDevice);
    cudaMemcpy(_opt_disp_image_gpu, _opt_disp_image, width * height * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(_init_p_gpu, &p[0], 6 * sizeof(float), cudaMemcpyHostToDevice);

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
    calNewDisp<<<blocks, threads>>>(subset, sideW, maxIter, width, height, _init_p_gpu, _origin_image_ref_gpu, _origin_image_target_gpu,
                                    _x_gradient_image, _y_gradient_image,
                                    _origin_disp_image_gpu, _opt_disp_image_gpu);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("calOptDisp time: %f ms\n", time);

    cudaMemcpy(_opt_disp_image, _opt_disp_image_gpu, width * height * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    cudaFree(_origin_disp_image_gpu);_origin_disp_image_gpu = nullptr;
    cudaFree(_opt_disp_image_gpu);_opt_disp_image_gpu = nullptr;
    cudaFree(_origin_image_ref_gpu);_origin_image_ref_gpu = nullptr;
    cudaFree(_origin_image_target_gpu);_origin_image_target_gpu = nullptr;
    cudaFree(_init_p_gpu);_init_p_gpu = nullptr;
}