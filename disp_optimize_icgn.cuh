#ifndef DISP_OPT_DISP_OPTIMIZE_ICGN_CUH
#define DISP_OPT_DISP_OPTIMIZE_ICGN_CUH
#include <opencv2/opencv.hpp>

class __declspec(dllexport) CDispOptimizeICGN_GPU {
public:
    CDispOptimizeICGN_GPU(){};
    ~CDispOptimizeICGN_GPU(){};
public:
    void run(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter, cv::Mat &_result);

private:
    void generate_gradient_image(cv::Mat &_l_image, cv::Mat &_x_gradient_image, cv::Mat &_y_gradient_image);

};


#endif //DISP_OPT_DISP_OPTIMIZE_ICGN_CUH
