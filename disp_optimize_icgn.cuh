#ifndef DISP_OPT_DISP_OPTIMIZE_ICGN_CUH
#define DISP_OPT_DISP_OPTIMIZE_ICGN_CUH
#include <opencv2/opencv.hpp>
class CDispOptimizeICGN_GPU {
public:
    CDispOptimizeICGN_GPU(){};
    ~CDispOptimizeICGN_GPU(){};
public:
    void run(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter, cv::Mat &_result);

private:

    //计算Sobel梯度影像;
    void generate_gradient_image(cv::Mat &_l_image, cv::Mat &_x_gradient_image, cv::Mat &_y_gradient_image);
    //计算Sobel梯度影像,输出为GPU显存;
    void generate_gradient_image(cv::Mat &_l_image, float *&_x_gradient_image, float *&_y_gradient_image);
    //计算每个像素的海森矩阵;
    void generate_hessian_mat(int subset, int sideW, int maxIter,int width, int height,float *_x_gradient_image, 
        float *_y_gradient_image, float *&_hessian_mat);
    //计算优化后的视差影像;
    void calOptDisp(int subset, int sideW, int maxIter,int width, int height,uchar *_origin_image_ref, uchar *_origin_image_target,
                    float *_x_gradient_image, float *_y_gradient_image, float *_mean_image, float *_sigma_image_gpu,
                    float *_hessian_inv_image,uchar *_origin_disp_image, float *_opt_disp_image);
    //计算均值影像;
    void calMeanImage(int subset, int sideW, int width, int height, uchar *_origin_image_ref,
                      float *&_mean_image, float *&_sigma_image);
    //计算海森矩阵的逆矩阵;
    void calInvHessianImage(int subset, int sideW, int width, int height,
                            float *_hessian_mat, float *&_hessian_mat_inv);
    //单独计算插值系数,经实验效果不好，暂时没有用单独计算的插值系数;
    void calInterpolationCoeff(int width, int height, uchar *_target_image, float *_x_gradient_image, float *_y_gradient_image, 
                                float *&_coeff_image);

};


#endif //DISP_OPT_DISP_OPTIMIZE_ICGN_CUH
