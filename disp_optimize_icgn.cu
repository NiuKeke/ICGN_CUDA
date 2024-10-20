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
#define N 6
__constant__ float MBT[4][4];
__constant__ float COEFF_A[16][16] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                      {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                      {-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                      {2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                      {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                                      {0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0},
                                      {0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0},
                                      {-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
                                      {0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0},
                                      {9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1},
                                      {-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1},
                                      {2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
                                      {0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0},
                                      {-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1},
                                      {4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1}};
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

__global__ void calHessianMat_kernel(int subset, int sideW, int width, int height, float *_x_grad_image,
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
                    int index =
                            (halfWinSize + threadIdx.y + j) * BLOCK_THREAD_DIM_Y * NUM_PER_THREAD_Y + halfWinSize + k +
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

__device__ void matrix_inverse6x6(float (*a)[N], float (*b)[N])
{
    using namespace std;
    int i, j, k;
    float max, temp;
    float t[N][N];
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            t[i][j] = a[i][j];
        }
    }
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            b[i][j] = (i == j) ? (float)1 : 0;
        }
    }

    for (i = 0; i < N; i++)
    {
        max = t[i][i];
        k = i;
        for (j = i + 1; j < N; j++)
        {
            if (fabsf(t[j][i]) > fabsf(max))
            {
                max = t[j][i];
                k = j;
            }
        }
        if (k != i)
        {
            for (j = 0; j < N; j++)
            {
                temp = t[i][j];
                t[i][j] = t[k][j];
                t[k][j] = temp;
                temp = b[i][j];
                b[i][j] = b[k][j];
                b[k][j] = temp;
            }
        }
        if (t[i][i] == 0)
        {
            break;
        }
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
    float det = a[0 * 3 + 0] * (a[1 * 3 + 1] * a[2 * 3 + 2] - a[2 * 3 + 1] * a[1 * 3 + 2]) -
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
    }
    else
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
                buffer = _src_image_sm[(halfWinSize + threadIdx.y + j) * 32 +
                                       halfWinSize + k + threadIdx.x];
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


__global__ void calOptDisp_kernel(int subset, int sideW, int maxIterNum, int width, int height,
                                  float *_init_p, uchar *_origin_image_ref, uchar *_origin_image_target,
                                  float *_x_grad_image, float *_y_grad_image, float *_mean_image,
                                  float *_sigma_image_gpu,
                                  float *_hessian_inv_image, uchar *_origin_disp,
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
    __shared__ float _target_value_intp_sm[16 * 16];                                      // 1k
    __shared__ float _target_sigma_value_intp_sm[16 * 16];                                // 1k;
    __shared__ float _target_delta_mean_value_intp_sm[16 * 16];                           // 1k
    __shared__ float _hessian_inv_image_sm[BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_X * 36]; // 9k;
    __shared__ float deltap_sm[6 * 16 * 16];                                              // 6k;
    __shared__ float dispValue_sm[64];                                                    // 0.25k;
    __shared__ int bFlags[16 * 16];
    bFlags[thread_index] = 1;
    if(thread_index < 64){
        dispValue_sm[thread_index] = 0.0f;
    }
    int block_index = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = 0; i < 6 * 6; i++)
    {
        int g_index = (halfWinSize)*width + halfWinSize + block_index * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y +
                      thread_index + i * width * height;
        if (thread_index < BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y)
        {
            _hessian_inv_image_sm[i * BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y +
                                  thread_index] = _hessian_inv_image[g_index];
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
        int iRow = i / BLOCK_THREAD_DIM_X + g_y + halfWinSize;
        int iCol = i % BLOCK_THREAD_DIM_X + g_x + halfWinSize;
        float delta_disp_value = 0.0f;
        int s_iRow = iRow - halfWinSize;
        int s_iCol = iCol - halfWinSize;
        int disp_value = _origin_disp[iRow * width + iCol];
        s_iCol -= disp_value;
        bool bRet = false;
        bRet = disp_value > 0;
        bRet &= (iRow - halfWinSize > 0);
        bRet &= (iRow + halfWinSize < height);
        bRet &= (iCol - halfWinSize > 0);
        bRet &= (iCol + halfWinSize < width);
        bRet &= (iCol - disp_value - halfWinSize > 0);
        bRet &= (iCol - disp_value + halfWinSize < width);

        int index = (thread_y - halfSubset + iRow) * width + thread_x - halfSubset + iCol;
        int grad_index = (thread_x - halfSubset + iRow) * width + thread_y - halfSubset + iCol;
        int sub_region_index = (iRow)*width + iCol;
        if (!bRet || index > width * height ||
            grad_index > width * height ||
            sub_region_index > width * height ||
            index < 0 || grad_index < 0 || sub_region_index < 0)
        {
            bRet = false;
        }

        uchar image_value_ref = _origin_image_ref[grad_index];
        float mean_image_value_ref = _mean_image[sub_region_index];
        float sigma_image_value_ref = _sigma_image_gpu[sub_region_index];
        float ref_delta_mean_value = image_value_ref - mean_image_value_ref;

        float Jacobian[6] = {0};
        Jacobian[0] = _x_grad_image[grad_index];
        Jacobian[3] = _y_grad_image[grad_index];

        float coord_weight[2] = {0};
        coord_weight[0] = float(thread_x - halfSubset) / float(halfSubset + 1);
        coord_weight[1] = float(thread_y - halfSubset) / float(halfSubset + 1);

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
                invHJacobian[jRow] += _hessian_inv_image_sm[tmp_index] * Jacobian[jCol];
            }
        }

        float thre = 1;
        int Iter = 0;
        float Czncc = 0;
        bool bValid = true;
        for (int iter_num = 0; iter_num < maxIterNum && thre > 1e-3 && bRet; ++iter_num)
        {

            float _target_value_intp = 0;
            _target_value_intp_sm[thread_y * 16 + thread_x] = 0;
            _target_sigma_value_intp_sm[thread_y * 16 + thread_x] = 0;
            _target_delta_mean_value_intp_sm[thread_y * 16 + thread_x] = 0;
            __syncthreads();
            float weight_x[4] = {0};
            float weight_y[4] = {0};
            int x_int = 0;
            int y_int = 0;
            int z_int = 0;
            int coor_x = thread_x - halfSubset;
            int coor_y = thread_y - halfSubset;
            float x = warP[0][0] * coor_x + warP[0][1] * coor_y + warP[0][2] * 1 + halfWinSize;
            float y = warP[1][0] * coor_x + warP[1][1] * coor_y + warP[1][2] * 1 + halfWinSize;
            float z = warP[2][0] * coor_x + warP[2][1] * coor_y + warP[2][2] * 1;

            x_int = floor(x);
            y_int = floor(y);
            z_int = floor(z);


            if ((s_iRow + x_int - 1) * width + y_int - 1 + s_iCol > width * height ||
                (s_iRow + x_int - 1) * width + y_int + s_iCol > width * height ||
                (s_iRow + x_int - 1) * width + y_int + 1 + s_iCol > width * height ||
                (s_iRow + x_int - 1) * width + y_int + 2 + s_iCol > width * height ||
                (s_iRow + x_int + 1 - 1) * width + y_int - 1 + s_iCol > width * height ||
                (s_iRow + x_int + 1 - 1) * width + y_int + s_iCol > width * height ||
                (s_iRow + x_int + 1 - 1) * width + y_int + 1 + s_iCol > width * height ||
                (s_iRow + x_int + 1 - 1) * width + y_int + 2 + s_iCol > width * height ||
                (s_iRow + x_int + 2 - 1) * width + y_int - 1 + s_iCol > width * height ||
                (s_iRow + x_int + 2 - 1) * width + y_int + s_iCol > width * height ||
                (s_iRow + x_int + 2 - 1) * width + y_int + 1 + s_iCol > width * height ||
                (s_iRow + x_int + 2 - 1) * width + y_int + 2 + s_iCol > width * height ||
                (s_iRow + x_int + 3 - 1) * width + y_int - 1 + s_iCol > width * height ||
                (s_iRow + x_int + 3 - 1) * width + y_int + s_iCol > width * height ||
                (s_iRow + x_int + 3 - 1) * width + y_int + 1 + s_iCol > width * height ||
                (s_iRow + x_int + 3 - 1) * width + y_int + 2 + s_iCol > width * height ||
                (s_iRow + x_int - 1) * width + y_int - 1 + s_iCol < 0 ||
                (s_iRow + x_int - 1) * width + y_int + s_iCol < 0 ||
                (s_iRow + x_int - 1) * width + y_int + 1 + s_iCol < 0 ||
                (s_iRow + x_int - 1) * width + y_int + 2 + s_iCol < 0 ||
                (s_iRow + x_int + 1 - 1) * width + y_int - 1 + s_iCol < 0 ||
                (s_iRow + x_int + 1 - 1) * width + y_int + s_iCol < 0 ||
                (s_iRow + x_int + 1 - 1) * width + y_int + 1 + s_iCol < 0 ||
                (s_iRow + x_int + 1 - 1) * width + y_int + 2 + s_iCol < 0 ||
                (s_iRow + x_int + 2 - 1) * width + y_int - 1 + s_iCol < 0 ||
                (s_iRow + x_int + 2 - 1) * width + y_int + s_iCol < 0 ||
                (s_iRow + x_int + 2 - 1) * width + y_int + 1 + s_iCol < 0 ||
                (s_iRow + x_int + 2 - 1) * width + y_int + 2 + s_iCol < 0 ||
                (s_iRow + x_int + 3 - 1) * width + y_int - 1 + s_iCol < 0 ||
                (s_iRow + x_int + 3 - 1) * width + y_int + s_iCol < 0 ||
                (s_iRow + x_int + 3 - 1) * width + y_int + 1 + s_iCol < 0 ||
                (s_iRow + x_int + 3 - 1) * width + y_int + 2 + s_iCol < 0 ||
                x < 2 || y < 2 || x > (width - 2) || y > (height - 2))
            {
                delta_disp_value = 0.0f;
                bValid = false;
            }

            {
                float delta_x = x - x_int;
                float delta_y = y - y_int;
                float delta_z = z - z_int;
                if (bValid && bRet)
                {
                    weight_x[0] =
                            MBT[0][0] * delta_x * delta_x * delta_x + MBT[0][1] * delta_x * delta_x + MBT[0][2] * delta_x + MBT[0][3];
                    weight_x[1] =
                            MBT[1][0] * delta_x * delta_x * delta_x + MBT[1][1] * delta_x * delta_x + MBT[1][2] * delta_x + MBT[1][3];
                    weight_x[2] =
                            MBT[2][0] * delta_x * delta_x * delta_x + MBT[2][1] * delta_x * delta_x + MBT[2][2] * delta_x + MBT[2][3];
                    weight_x[3] =
                            MBT[3][0] * delta_x * delta_x * delta_x + MBT[3][1] * delta_x * delta_x + MBT[3][2] * delta_x + MBT[3][3];

                    weight_y[0] =
                            MBT[0][0] * delta_y * delta_y * delta_y + MBT[0][1] * delta_y * delta_y + MBT[0][2] * delta_y + MBT[0][3];
                    weight_y[1] =
                            MBT[1][0] * delta_y * delta_y * delta_y + MBT[1][1] * delta_y * delta_y + MBT[1][2] * delta_y + MBT[1][3];
                    weight_y[2] =
                            MBT[2][0] * delta_y * delta_y * delta_y + MBT[2][1] * delta_y * delta_y + MBT[2][2] * delta_y + MBT[2][3];
                    weight_y[3] =
                            MBT[3][0] * delta_y * delta_y * delta_y + MBT[3][1] * delta_y * delta_y + MBT[3][2] * delta_y + MBT[3][3];

                    float target_value_intp[4][4] = {0};
                    if (thread_x < subset && thread_y < subset)
                    {
                        target_value_intp[0][0] = _origin_image_target[(s_iRow + x_int - 1)     * width + y_int - 1 + s_iCol];
                        target_value_intp[0][1] = _origin_image_target[(s_iRow + x_int - 1)     * width + y_int     + s_iCol];
                        target_value_intp[0][2] = _origin_image_target[(s_iRow + x_int - 1)     * width + y_int + 1 + s_iCol];
                        target_value_intp[0][3] = _origin_image_target[(s_iRow + x_int - 1)     * width + y_int + 2 + s_iCol];
                        target_value_intp[1][0] = _origin_image_target[(s_iRow + x_int + 1 - 1) * width + y_int - 1 + s_iCol];
                        target_value_intp[1][1] = _origin_image_target[(s_iRow + x_int + 1 - 1) * width + y_int     + s_iCol];
                        target_value_intp[1][2] = _origin_image_target[(s_iRow + x_int + 1 - 1) * width + y_int + 1 + s_iCol];
                        target_value_intp[1][3] = _origin_image_target[(s_iRow + x_int + 1 - 1) * width + y_int + 2 + s_iCol];
                        target_value_intp[2][0] = _origin_image_target[(s_iRow + x_int + 2 - 1) * width + y_int - 1 + s_iCol];
                        target_value_intp[2][1] = _origin_image_target[(s_iRow + x_int + 2 - 1) * width + y_int     + s_iCol];
                        target_value_intp[2][2] = _origin_image_target[(s_iRow + x_int + 2 - 1) * width + y_int + 1 + s_iCol];
                        target_value_intp[2][3] = _origin_image_target[(s_iRow + x_int + 2 - 1) * width + y_int + 2 + s_iCol];
                        target_value_intp[3][0] = _origin_image_target[(s_iRow + x_int + 3 - 1) * width + y_int - 1 + s_iCol];
                        target_value_intp[3][1] = _origin_image_target[(s_iRow + x_int + 3 - 1) * width + y_int     + s_iCol];
                        target_value_intp[3][2] = _origin_image_target[(s_iRow + x_int + 3 - 1) * width + y_int + 1 + s_iCol];
                        target_value_intp[3][3] = _origin_image_target[(s_iRow + x_int + 3 - 1) * width + y_int + 2 + s_iCol];
                    }

                    _target_value_intp += target_value_intp[0][0] * weight_y[0] * weight_x[0];
                    _target_value_intp += target_value_intp[0][1] * weight_y[1] * weight_x[0];
                    _target_value_intp += target_value_intp[0][2] * weight_y[2] * weight_x[0];
                    _target_value_intp += target_value_intp[0][3] * weight_y[3] * weight_x[0];
                    _target_value_intp += target_value_intp[1][0] * weight_y[0] * weight_x[1];
                    _target_value_intp += target_value_intp[1][1] * weight_y[1] * weight_x[1];
                    _target_value_intp += target_value_intp[1][2] * weight_y[2] * weight_x[1];
                    _target_value_intp += target_value_intp[1][3] * weight_y[3] * weight_x[1];
                    _target_value_intp += target_value_intp[2][0] * weight_y[0] * weight_x[2];
                    _target_value_intp += target_value_intp[2][1] * weight_y[1] * weight_x[2];
                    _target_value_intp += target_value_intp[2][2] * weight_y[2] * weight_x[2];
                    _target_value_intp += target_value_intp[2][3] * weight_y[3] * weight_x[2];
                    _target_value_intp += target_value_intp[3][0] * weight_y[0] * weight_x[3];
                    _target_value_intp += target_value_intp[3][1] * weight_y[1] * weight_x[3];
                    _target_value_intp += target_value_intp[3][2] * weight_y[2] * weight_x[3];
                    _target_value_intp += target_value_intp[3][3] * weight_y[3] * weight_x[3];

                    _target_value_intp_sm[thread_y * 16 + thread_x] = _target_value_intp;
                    _target_sigma_value_intp_sm[thread_y * 16 + thread_x] = _target_value_intp * _target_value_intp;
                    _target_delta_mean_value_intp_sm[thread_y * 16 + thread_x] = _target_value_intp;
                }


                __syncthreads();

                for (int c = 16 / 2; c > 0; c >>= 1)
                {
                    if (thread_x < c)
                    {
                        _target_value_intp_sm[thread_y * 16 + thread_x] += _target_value_intp_sm[(thread_y) * 16 +
                                                                                                 (thread_x + c)];
                        _target_sigma_value_intp_sm[thread_y * 16 + thread_x] += _target_sigma_value_intp_sm[(thread_y) * 16 + (thread_x + c)];
                    }
                    __syncthreads();
                }

                for (int r = 16 / 2; r > 0; r >>= 1)
                {
                    if (thread_y < r)
                    {
                        _target_value_intp_sm[thread_y * 16 + thread_x] += _target_value_intp_sm[(thread_y + r) * 16 +
                                                                                                 (thread_x)];
                        _target_sigma_value_intp_sm[thread_y * 16 + thread_x] += _target_sigma_value_intp_sm[(thread_y + r) * 16 + (thread_x)];
                    }
                    __syncthreads();
                }

                __syncthreads();
                float target_value_sum = _target_value_intp_sm[0];
                float target_mean_value = target_value_sum / float(subset * subset);
                float target_delta_squar = _target_sigma_value_intp_sm[0] - target_value_sum * target_mean_value;
                target_delta_squar = sqrt(target_delta_squar);
                _target_delta_mean_value_intp_sm[thread_y * 16 + thread_x] -= target_mean_value;
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
                if (thread_x < subset && thread_y < subset)
                {
                    deltap[0] = -invHJacobian[0] * tmp;
                    deltap[1] = -invHJacobian[1] * tmp;
                    deltap[2] = -invHJacobian[2] * tmp;
                    deltap[3] = -invHJacobian[3] * tmp;
                    deltap[4] = -invHJacobian[4] * tmp;
                    deltap[5] = -invHJacobian[5] * tmp;
                }

                deltap_sm[thread_index + 0 * 16 * 16] = deltap[0];
                deltap_sm[thread_index + 1 * 16 * 16] = deltap[1];
                deltap_sm[thread_index + 2 * 16 * 16] = deltap[2];
                deltap_sm[thread_index + 3 * 16 * 16] = deltap[3];
                deltap_sm[thread_index + 4 * 16 * 16] = deltap[4];
                deltap_sm[thread_index + 5 * 16 * 16] = deltap[5];

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
                if (thread_index == 0)
                {
                    float M[6] = {1, 1.0f / subset, 1.0f / subset, 1, 1.0f / subset, 1.0f / subset};
                    deltap_sm[0 * 16 * 16] *= M[0];
                    deltap_sm[1 * 16 * 16] *= M[1];
                    deltap_sm[2 * 16 * 16] *= M[2];
                    deltap_sm[3 * 16 * 16] *= M[3];
                    deltap_sm[4 * 16 * 16] *= M[4];
                    deltap_sm[5 * 16 * 16] *= M[5];
                }
                __syncthreads();

                float warP2[3][3] = {0};
                float delta_warp_p[3 * 3] = {1 + deltap_sm[1 * 16 * 16], deltap_sm[2 * 16 * 16], deltap_sm[0 * 16 * 16],
                                             deltap_sm[4 * 16 * 16], 1 + deltap_sm[5 * 16 * 16], deltap_sm[3 * 16 * 16],
                                             0, 0, 1};
                float invwarpdelta[3][3] = {0};
                if (bValid && bRet)
                {
                    inverse3x3(delta_warp_p, invwarpdelta);
                    warP2[0][0] = warP[0][0] * invwarpdelta[0][0] + warP[0][1] * invwarpdelta[1][0] + warP[0][2] * invwarpdelta[2][0];
                    warP2[0][1] = warP[0][0] * invwarpdelta[0][1] + warP[0][1] * invwarpdelta[1][1] + warP[0][2] * invwarpdelta[2][1];
                    warP2[0][2] = warP[0][0] * invwarpdelta[0][2] + warP[0][1] * invwarpdelta[1][2] + warP[0][2] * invwarpdelta[2][2];
                    warP2[1][0] = warP[1][0] * invwarpdelta[0][0] + warP[1][1] * invwarpdelta[1][0] + warP[1][2] * invwarpdelta[2][0];
                    warP2[1][1] = warP[1][0] * invwarpdelta[0][1] + warP[1][1] * invwarpdelta[1][1] + warP[1][2] * invwarpdelta[2][1];
                    warP2[1][2] = warP[1][0] * invwarpdelta[0][2] + warP[1][1] * invwarpdelta[1][2] + warP[1][2] * invwarpdelta[2][2];
                    warP2[2][0] = warP[2][0] * invwarpdelta[0][0] + warP[2][1] * invwarpdelta[1][0] + warP[2][2] * invwarpdelta[2][0];
                    warP2[2][1] = warP[2][0] * invwarpdelta[0][1] + warP[2][1] * invwarpdelta[1][1] + warP[2][2] * invwarpdelta[2][1];
                    warP2[2][2] = warP[2][0] * invwarpdelta[0][2] + warP[2][1] * invwarpdelta[1][2] + warP[2][2] * invwarpdelta[2][2];
                }
                float delta_value = deltap_sm[0 * 16 * 16] * deltap_sm[0 * 16 * 16] +
                                    deltap_sm[3 * 16 * 16] * deltap_sm[3 * 16 * 16];
                thre = sqrt(delta_value);

                __syncthreads();
                if (bValid && bRet)
                {
                    warP[0][0] = warP2[0][0];
                    warP[0][1] = warP2[0][1];
                    warP[0][2] = warP2[0][2];
                    warP[1][0] = warP2[1][0];
                    warP[1][1] = warP2[1][1];
                    warP[1][2] = warP2[1][2];
                    warP[2][0] = warP2[2][0];
                    warP[2][1] = warP2[2][1];
                    warP[2][2] = warP2[2][2];
                    delta_disp_value = warP[1][2];
                }

                __syncthreads();
            }
        }
        __syncthreads();

        bFlags[thread_index] = bValid && bRet;
        if (bValid && bRet && thread_index == 0) {
            for(int n = 0; n < 16 * 16; n++){
                bValid &= bFlags[n] != 0;
            }
            if(bValid && !isnan(delta_disp_value)){
                dispValue_sm[i] = delta_disp_value;
            }

        }
        __syncthreads();
    }
    __syncthreads();
    if (thread_index < BLOCK_THREAD_DIM_X * BLOCK_THREAD_DIM_Y)
    {
        int iRow = thread_index / BLOCK_THREAD_DIM_X + g_y + halfWinSize;
        int iCol = thread_index % BLOCK_THREAD_DIM_X + g_x + halfWinSize;
        int disp_index = iRow * width + iCol;
        if (disp_index < width * height && disp_index >= 0)
        {
            _opt_disp[disp_index] = _origin_disp[disp_index] + dispValue_sm[thread_index];
        }
    }
}

__global__ void calInterpolationCoeff_kernel(int width, int height, uchar *_target_image,
                                             float *_x_gradient_image, float *_y_gradient_image,
                                             float *_coeff_image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    if (x >= width || y >= height || (y + 1) >= height || (x + 1) >= width)
    {
        return;
    }
    float belta[16];
    belta[0] = _target_image[y * width + x];
    belta[1] = _target_image[y * width + x + 1];
    belta[2] = _target_image[(y + 1) * width + x];
    belta[3] = _target_image[(y + 1) * width + x + 1];

    belta[4] = _x_gradient_image[y * width + x];
    belta[5] = _x_gradient_image[y * width + x + 1];
    belta[6] = _x_gradient_image[(y + 1) * width + x];
    belta[7] = _x_gradient_image[(y + 1) * width + x + 1];

    belta[8] = _y_gradient_image[y * width + x];
    belta[9] = _y_gradient_image[y * width + x + 1];
    belta[10] = _y_gradient_image[(y + 1) * width + x];
    belta[11] = _y_gradient_image[(y + 1) * width + x + 1];

    belta[12] = sqrt(belta[4] * belta[4] + belta[8] * belta[8]);
    belta[13] = sqrt(belta[5] * belta[5] + belta[9] * belta[9]);
    belta[14] = sqrt(belta[6] * belta[6] + belta[10] * belta[10]);
    belta[15] = sqrt(belta[7] * belta[7] + belta[11] * belta[11]);

    float result[16] = {0};
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            result[i] += COEFF_A[i][j] * belta[j];
        }
    }

    __syncthreads();

    for (int i = 0; i < 16; i++)
    {
        _coeff_image[i * width * height + index] = result[i];
    }
}

void CDispOptimizeICGN_GPU::run(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW,
                                int maxIter,
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
#ifdef _DEBUG
    {
        cudaMemcpy(_x_gradient_image_cpu.data, _x_gradient_image, _l_image.rows * _l_image.cols * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(_y_gradient_image_cpu.data, _y_gradient_image, _l_image.rows * _l_image.cols * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cv::imwrite("x_gradient_image_cpu.tif", _x_gradient_image_cpu);
        cv::imwrite("y_gradient_image_cpu.tif", _y_gradient_image_cpu);
    }
#endif
    // 生成右图像梯度影像;
    // float *_target_x_gradient_image = nullptr;
    // float *_target_y_gradient_image = nullptr;
    // generate_gradient_image(_r_image, _target_x_gradient_image, _target_y_gradient_image);

    // float *_target_interpolation_coeff = nullptr;
    // calInterpolationCoeff(_r_image.cols, _r_image.rows, _r_image.data,
    //                       _target_x_gradient_image, _target_y_gradient_image, _target_interpolation_coeff);

#ifdef _DBEUG
    {
        cv::Mat _target_interpolation_coeff_cpu = cv::Mat(16, _r_image.cols * _r_image.rows, CV_32FC1);
        cudaMemcpy(_target_interpolation_coeff_cpu.data, _target_interpolation_coeff,
                   _r_image.rows * _r_image.cols * sizeof(float) * 16,
                   cudaMemcpyDeviceToHost);
        cv::imwrite("target_interpolation_coeff_cpu.tif", _target_interpolation_coeff_cpu);
    }
#endif
    float *hessian_mat = nullptr;
    generate_hessian_mat(subset, sideW, maxIter, _l_image.cols, _l_image.rows, _x_gradient_image, _y_gradient_image,
                         hessian_mat);
#ifdef _DEBUG
    {
        cv::Mat hessian = cv::Mat(36, _l_image.cols * _l_image.rows, CV_32FC1);
        cudaMemcpy(hessian.data, hessian_mat, _l_image.cols * _l_image.rows * 36 * sizeof(float),
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);

        cv::imwrite("./hessian.tif", hessian);
    }
#endif
    // 计算均值影像;

    float *_mean_image_gpu = nullptr;
    float *_sigma_image_gpu = nullptr;
    calMeanImage(subset, sideW, _l_image.cols, _l_image.rows, (uchar *)_l_image.data, _mean_image_gpu,
                 _sigma_image_gpu);
#ifdef _DEBUG
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
#endif
    float *hessian_inv = nullptr;
    calInvHessianImage(subset, sideW, _l_image.cols, _l_image.rows, hessian_mat, hessian_inv);
#ifdef _DEBUG
    {
        cv::Mat hessian_inv_image = cv::Mat(36, _l_image.cols * _l_image.rows, CV_32FC1);
        cudaMemcpy((float *)hessian_inv_image.data, hessian_inv, 36 * sizeof(float) * _l_image.cols * _l_image.rows,
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);

        cv::imwrite("./hessian_inv_image.tif", hessian_inv_image);
    }
#endif
    calOptDisp(subset, sideW, maxIter, _l_image.cols, _r_image.rows, _l_image.data, _r_image.data,
               _x_gradient_image, _y_gradient_image, _mean_image_gpu, _sigma_image_gpu, hessian_inv,
               (uchar *)_src_disp.data, (float *)_result.data);
#ifdef _DEBUG
    {
        cv::imwrite("./disp_result.tif", _result);
    }
#endif
    cudaFree(_x_gradient_image);_x_gradient_image = nullptr;
    cudaFree(_y_gradient_image);_y_gradient_image = nullptr;
    // cudaFree(_target_x_gradient_image);_target_x_gradient_image = nullptr;
    // cudaFree(_target_y_gradient_image);_target_y_gradient_image = nullptr;
    // cudaFree(_target_interpolation_coeff);_target_interpolation_coeff = nullptr;
    cudaFree(hessian_mat);hessian_mat = nullptr;
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
    calHessianMat_kernel<<<blocks, threads>>>(subset, sideW, width, height, _x_gradient_image,
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

    cudaFree(src_image);
    src_image = nullptr;
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
                                       float *_y_gradient_image, float *_mean_image, float *_sigma_image_gpu,
                                       float *_hessian_inv_image,
                                       uchar *_origin_disp_image, float *_opt_disp_image)
{
    uchar *_origin_disp_image_gpu = nullptr;
    float *_opt_disp_image_gpu = nullptr;
    uchar *_origin_image_ref_gpu = nullptr;
    uchar *_origin_image_target_gpu = nullptr;
    float p[6] = {0.4686, 0, 0, -0.2116, 0, 0};
    float *_init_p_gpu = nullptr;
    cudaMalloc((void **)&_origin_disp_image_gpu, width * height * sizeof(uchar));
    cudaMalloc((void **)&_opt_disp_image_gpu, width * height * sizeof(float));
    cudaMalloc((void **)&_origin_image_ref_gpu, width * height * sizeof(uchar));
    cudaMalloc((void **)&_origin_image_target_gpu, width * height * sizeof(uchar));
    cudaMalloc((void **)&_init_p_gpu, 6 * sizeof(float));

    cudaMemcpy(_origin_disp_image_gpu, _origin_disp_image, width * height * sizeof(uchar),
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

    dim3 threads(16, 16);
    dim3 blocks((width - 2 * halfWinSize + BLOCK_THREAD_DIM_X - 1) / (BLOCK_THREAD_DIM_X),
                (height - 2 * halfWinSize + BLOCK_THREAD_DIM_X - 1) / (BLOCK_THREAD_DIM_X));
    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    calOptDisp_kernel<<<blocks, threads>>>(subset, sideW, maxIter, width, height, _init_p_gpu,
                                           _origin_image_ref_gpu,
                                           _origin_image_target_gpu,
                                           _x_gradient_image, _y_gradient_image, _mean_image,
                                           _sigma_image_gpu,
                                           _hessian_inv_image, _origin_disp_image_gpu, _opt_disp_image_gpu);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("calOptDisp time: %f ms\n", time);

    cudaMemcpy(_opt_disp_image, _opt_disp_image_gpu, width * height * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(_origin_disp_image_gpu); _origin_disp_image_gpu = nullptr;
    cudaFree(_opt_disp_image_gpu);_opt_disp_image_gpu = nullptr;
    cudaFree(_origin_image_ref_gpu);_origin_image_ref_gpu = nullptr;
    cudaFree(_origin_image_target_gpu);_origin_image_target_gpu = nullptr;
    cudaFree(_init_p_gpu);_init_p_gpu = nullptr;
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

    cudaFree(_src_image_gpu);
    _src_image_gpu = nullptr;

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

void CDispOptimizeICGN_GPU::calInterpolationCoeff(int width, int height, uchar *_target_image,
                                                  float *_x_gradient_image, float *_y_gradient_image,
                                                  float *&_coeff_image)
{
    cudaMalloc((void **)&_coeff_image, sizeof(float) * width * height * 16);
    uchar *_target_image_gpu = nullptr;
    cudaMalloc((void **)&_target_image_gpu, sizeof(uchar) * width * height);
    cudaMemcpy(_target_image_gpu, _target_image, sizeof(uchar) * width * height, cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    cudaEvent_t start, stop;
    float time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    calInterpolationCoeff_kernel<<<blocks, threads>>>(width, height, _target_image_gpu, _x_gradient_image, _y_gradient_image,
                                                      _coeff_image);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    printf("calInterpolationCoeff time: %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(_target_image_gpu);_target_image_gpu = nullptr;
}