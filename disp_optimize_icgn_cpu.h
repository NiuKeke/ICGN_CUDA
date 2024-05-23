//
// Created by Administrator on 2024/4/28.
//

#ifndef DISP_OPT_DISP_OPTIMIZE_ICGN_CPU_H
#define DISP_OPT_DISP_OPTIMIZE_ICGN_CPU_H
#include <opencv2/opencv.hpp>

class  CDispOptimizeICGN_CPU {
public:
    CDispOptimizeICGN_CPU(){};
    ~CDispOptimizeICGN_CPU(){};
public:
    void run(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter, cv::Mat &_result);
    void run_old(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter,
                                        cv::Mat &_result);
private:
    void generate_gradient_image(cv::Mat &_l_image, cv::Mat &_x_gradient_image, cv::Mat &_y_gradient_image);

    int calLocalSubHom(int subset, int *localSubHom, float *localSub);

    int ICGNalgo(unsigned char *srcL, unsigned char *srcR, float *gradxImg, float *gradyImg, int *localSubHom,
             float *localSub,
                 uchar *disp, double *dispFloat, int height, int width, int subset, int sideW, int maxIter);

    int IterICGN2(unsigned char *ImRef, unsigned char *ImDef, float *gradxImgRef, float *gradyImgRef, int *localSubHom,
                  float *localSub, float *p, int *pCoord, double *dispBias, int maxIter, int subset, int halfSubset,
                  int sideW, int length);
    int GetMatrixInverse(double src[3][3], int n, double des[3][3]);
    double getA(double arcs[3][3], int n);
    int  getAStart6(double arcs[6][6], int n, double ans[6][6]);
    int  getAStart(double arcs[3][3], int n, double ans[3][3]);
    double getA6(double arcs[6][6], int n);
    int GetMatrixInverse6(double src[6][6], int n, double des[6][6]);



    int calLocalSubHom_cv(int subset, cv::Mat &_localSubHom, cv::Mat &_localSub);
    int ICGNalgo_new(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, cv::Mat &gradImage_x,
                     cv::Mat &gradImage_y, cv::Mat &localSubHom, cv::Mat &localSub, cv::Mat &opt_disp,
                     int subset, int sideW, int maxIter);

    int CalOptDisp(cv::Mat &ImRef, cv::Mat &ImDef, cv::Mat &gradxImgRef, cv::Mat &gradyImgRef,
                   cv::Mat &localSubHom, cv::Mat &localSub, float p[], int pCoord[], double* dispBias, int maxIter,
                   int subset, int halfSubset, int sideW, int length);

    //
    int calHessian(int row, int col, unsigned char *ImRef, unsigned char *ImDef, float *gradxImgRef, float *gradyImgRef,
                                     int *localSubHom, float *localSub, float p[], int pCoord[], double *dispBias, int maxIter,
                                     int subset, int halfSubset, int sideW, int length, double (*H)[6], bool bDebug);
};


#endif //DISP_OPT_DISP_OPTIMIZE_ICGN_CPU_H
