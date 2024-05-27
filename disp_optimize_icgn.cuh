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
    void generate_gradient_image(cv::Mat &_l_image, cv::Mat &_x_gradient_image, cv::Mat &_y_gradient_image);
    //
    void generate_gradient_image(cv::Mat &_l_image, float *&_x_gradient_image, float *&_y_gradient_image);
    //
    void generate_hessian_mat(int subset, int sideW, int maxIter,int width, int height,float *_x_gradient_image, 
        float *_y_gradient_image, double *&_hessian_mat);
    //
    void calOptDisp(int subset, int sideW, int maxIter,int width, int height,uchar *_origin_image_ref, uchar *_origin_image_target,
                    float *_x_gradient_image, float *_y_gradient_image, float *_origin_disp_image, float *_opt_disp_image);

};


#endif //DISP_OPT_DISP_OPTIMIZE_ICGN_CUH
