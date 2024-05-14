#include "disp_optimize_icgn.cuh"
#include <opencv2/opencv.hpp>
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

__global__ void calHessianMat(int subset, int sideW, int width, int height, float *_x_grad_image, float *_y_grad_image, float *_hessian_mat)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW;//7+5;

    if((x - halfWinSize) < 0 || (x + halfWinSize) > width || (y - halfWinSize) < 0 || (y + halfWinSize) > height){
        return ;
    }

    
}
void CDispOptimizeICGN_GPU::run(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter, cv::Mat &_result)
{
    // 生成左图像梯度影像,分为x,y两个方向;
    cv::Mat _x_gradient_image, _y_gradient_image;
    _x_gradient_image.create(_l_image.size(), CV_32FC1);
    _y_gradient_image.create(_l_image.size(), CV_32FC1);
    generate_gradient_image(_l_image, _x_gradient_image, _y_gradient_image);
    // 保存梯度影像;
    cv::imwrite("x_gradient_image_gpu.jpg", _x_gradient_image);
    cv::imwrite("y_gradient_image_gpu.jpg", _y_gradient_image);
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