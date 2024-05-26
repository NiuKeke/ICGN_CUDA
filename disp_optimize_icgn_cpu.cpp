//
// Created by Administrator on 2024/4/28.
//

#include "disp_optimize_icgn_cpu.h"
void CDispOptimizeICGN_CPU::run_old(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter,
                                    cv::Mat &_result)
{

    // 生成左图像梯度影像,分为x,y两个方向;
    cv::Mat _x_gradient_image, _y_gradient_image;
    _x_gradient_image.create(_l_image.size(), CV_32FC1);
    _y_gradient_image.create(_l_image.size(), CV_32FC1);
    generate_gradient_image(_l_image, _x_gradient_image, _y_gradient_image);
    // 保存梯度影像;
    cv::imwrite("x_gradient_image.tif", _x_gradient_image);
    cv::imwrite("y_gradient_image.tif", _y_gradient_image);

    int *localSubHom = new int[3 * subset * subset];
    float *localSub = new float[2 * subset * subset];
    memset(localSubHom, 0, sizeof(int) * 3 * subset * subset);
    memset(localSub, 0, sizeof(float) * 2 * subset * subset);
    calLocalSubHom(subset, localSubHom, localSub);

    ICGNalgo(_l_image.data, _r_image.data, (float *)_x_gradient_image.data, (float *)_y_gradient_image.data,
             localSubHom, localSub, _src_disp.data, (double *)_result.data,
             _l_image.rows, _l_image.cols, subset, sideW, maxIter);
    delete[] localSubHom;
    delete[] localSub;
}
void CDispOptimizeICGN_CPU::run(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, int subset, int sideW, int maxIter,
                                cv::Mat &_result)
{

    // 生成左图像梯度影像,分为x,y两个方向;
    cv::Mat _x_gradient_image, _y_gradient_image;
    _x_gradient_image.create(_l_image.size(), CV_32FC1);
    _y_gradient_image.create(_l_image.size(), CV_32FC1);
    generate_gradient_image(_l_image, _x_gradient_image, _y_gradient_image);
    // 保存梯度影像;
    cv::imwrite("x_gradient_image.jpg", _x_gradient_image);
    cv::imwrite("y_gradient_image.jpg", _y_gradient_image);

    cv::Mat localSubHom = cv::Mat(subset, subset, CV_32FC3);
    localSubHom.setTo(0);
    cv::Mat localSub = cv::Mat(subset, subset, CV_32FC2);
    localSub.setTo(0);
    calLocalSubHom_cv(subset, localSubHom, localSub);
}

void CDispOptimizeICGN_CPU::generate_gradient_image(cv::Mat &_l_image, cv::Mat &_x_gradient_image,
                                                    cv::Mat &_y_gradient_image)
{
    int height = _l_image.rows;
    int width = _l_image.cols;
    for (int i = 0; i < height * width; i++)
    {
        int r = i / width;
        int c = i % width;

        uchar *ImRef = _l_image.data;
        if (c >= 2 && c < width - 2)
        {
            float result = 0.0f;
            result -= ImRef[r * width + c + 2] * 0.083333333333333;
            result += ImRef[r * width + c + 1] * 0.666666666666667;
            result -= ImRef[r * width + c - 1] * 0.666666666666667;
            result += ImRef[r * width + c - 2] * 0.083333333333333;
            *_y_gradient_image.ptr<float>(r, c) = result;
        }
        else
        {
            *_y_gradient_image.ptr<float>(r, c) = 0;
        }
        if (r >= 2 && r < height - 2)
        {
            float result = 0.0f;
            result -= ImRef[(r + 2) * width + c] * 0.083333333333333;
            result += ImRef[(r + 1) * width + c] * 0.666666666666667;
            result -= ImRef[(r - 1) * width + c] * 0.666666666666667;
            result += ImRef[(r - 2) * width + c] * 0.083333333333333;
            *_x_gradient_image.ptr<float>(r, c) = result;
        }
        else
        {
            *_x_gradient_image.ptr<float>(r, c) = 0;
        }
    }
}

int CDispOptimizeICGN_CPU::calLocalSubHom_cv(int subset, cv::Mat &_localSubHom, cv::Mat &_localSub)
{
    cv::Mat tmp_channel_0 = cv::Mat(_localSubHom.rows, _localSubHom.cols, CV_32FC1);
    int halfSubset = subset / 2;
    for (int iRow = 0; iRow < subset; ++iRow)
    {
        float value = -(float)halfSubset + (float)iRow;
        printf("value : %lf\n", value);
        for (int iCol = 0; iCol < subset; ++iCol)
        {
            tmp_channel_0.at<float>(iRow, iCol) = value;
        }
    }

    cv::Mat tmp_channel_1 = cv::Mat(_localSubHom.rows, _localSubHom.cols, CV_32FC1);
    tmp_channel_1 = tmp_channel_0.t();

    cv::Mat tmp_channel_2 = cv::Mat(_localSubHom.rows, _localSubHom.cols, CV_32FC1);
    tmp_channel_2.setTo(1);
    cv::imwrite("tmp_channel_0.tif", tmp_channel_0);
    cv::imwrite("tmp_channel_1.tif", tmp_channel_1);
    cv::imwrite("tmp_channel_2.tif", tmp_channel_2);

    cv::merge(std::vector<cv::Mat>{tmp_channel_0, tmp_channel_1, tmp_channel_2}, _localSubHom);

    cv::Mat _localSub_tmp0 = cv::Mat(_localSub.rows, _localSub.cols, CV_32FC1);
    _localSub_tmp0 = tmp_channel_0 / double(halfSubset + 1.0f);
    cv::Mat _localSub_tmp1 = cv::Mat(_localSub.rows, _localSub.cols, CV_32FC1);
    _localSub_tmp1 = tmp_channel_1 / double(halfSubset + 1.0f);

    cv::imwrite("_localSub_tmp0.tif", _localSub_tmp0);
    cv::imwrite("_localSub_tmp1.tif", _localSub_tmp1);

    cv::merge(std::vector<cv::Mat>{_localSub_tmp0, _localSub_tmp1}, _localSub);

    return 0;
}
int CDispOptimizeICGN_CPU::calLocalSubHom(int subset, int *localSubHom, float *localSub)
{
    int halfSubset = subset / 2;
    for (int ii = 0; ii < subset * subset; ii++)
    {
        if (ii % subset == 0)
        {
            localSubHom[ii] = -halfSubset;
        }
        else
        {
            localSubHom[ii] = localSubHom[ii - 1] + 1;
        }
        if (ii == 0)
        {
            localSubHom[ii + subset * subset] = -halfSubset;
        }
        else if (ii > 0 && ii % subset == 0)
        {
            localSubHom[ii + subset * subset] = localSubHom[ii + subset * subset - 1] + 1;
        }
        else
        {
            localSubHom[ii + subset * subset] = localSubHom[ii + subset * subset - 1];
        }

        localSubHom[ii + 2 * subset * subset] = 1;
        localSub[ii] = localSubHom[ii] / float(halfSubset + 1);
        localSub[ii + subset * subset] = localSubHom[ii + subset * subset] / float(halfSubset + 1);
    }

    // for (int i = 0; i < 3; ++i)
    // {
    //     for (int j = 0; j < subset; ++j)
    //     {
    //         for (int k = 0; k < subset; ++k)
    //         {
    //             printf("channel: %d, col:%d, row: %d, localSubHom[%d]: %d\n",
    //                    i, j, k, i * subset * subset + j * subset + k, localSubHom[i * subset * subset + j * subset + k]);
    //         }
    //     }
    // }
    // for (int i = 0; i < 2; ++i)
    // {
    //     for (int j = 0; j < subset; ++j)
    //     {
    //         for (int k = 0; k < subset; ++k)
    //         {
    //             printf("channel: %d, col:%d, row: %d, localSub[%d]: %f\n",
    //                    i, j, k, i * subset * subset + j * subset + k, localSub[i * subset * subset + j * subset + k]);
    //         }
    //     }
    // }
    return 1;
}

int CDispOptimizeICGN_CPU::ICGNalgo(unsigned char *srcL, unsigned char *srcR, float *gradxImg, float *gradyImg,
                                    int *localSubHom, float *localSub, uchar *disp, double *dispFloat,
                                    int height, int width, int subset, int sideW, int maxIter)
{
    int ImgSize = height * width;
    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int pCoord[3] = {halfWinSize, halfWinSize, 1};
    int length = 2 * halfWinSize + 1; // 25=2*12 + 1;
    int sizeX = subset + 2 * sideW;   // 15+2*5=25;
    int sizeY = subset + 2 * sideW;
    cv::Mat hessianMat = cv::Mat(36, height * width, CV_64FC1);
    for (int i = 0; i < ImgSize; i++)
    {
        int irow = i / width;
        int icol = i % width;
        bool bRet = false;
        bRet = disp[i] > 0;
        bRet &= (irow - halfWinSize > 0);
        bRet &= (irow + halfWinSize < height);
        bRet &= (icol - halfWinSize > 0);
        bRet &= (icol + halfWinSize < width);
        bRet &= (icol - disp[i] - halfWinSize > 0);
        bRet &= (icol - disp[i] + halfWinSize < width);

        if (bRet)
        {
            unsigned char *ImRef = new unsigned char[length * length];
            unsigned char *ImDef = new unsigned char[length * length];
            float *gradxImgRef = new float[length * length];
            float *gradyImgRef = new float[length * length];
            int num = 0;
            for (int j = -halfWinSize; j <= halfWinSize; j++)
            {
                for (int k = -halfWinSize; k <= halfWinSize; k++)
                {
                    ImRef[num] = srcL[i + j * width + k];
                    ImDef[num] = srcR[i + j * width + k - disp[i]];
                    gradxImgRef[num] = gradxImg[i + j * width + k];
                    gradyImgRef[num] = gradyImg[i + j * width + k];
                    if(i == 16961){
                        // printf("row: %d, col: %d, gradxImg[%d]: %lf, gradyImg[%d]: %lf, srcL[%d]: %d, srcR[%d]: %d\n",
                        // irow + j,icol + k,
                        // i + j * width + k, gradxImg[i + j * width + k],
                        // i + j * width + k, gradyImg[i + j * width + k],
                        // i + j * width + k, srcL[i + j * width + k],
                        // i + j * width + k - disp[i], srcR[i + j * width + k - disp[i]]);
                    }
                    num++;
                }
            }
            // std::cout << "row:             " << irow << std::endl;
            // std::cout << "col:             " << icol << std::endl;
            float p[6] = {0.4686, 0, 0, -0.2116, 0, 0};
            double *dispBias = new double[2];
            memset(dispBias, 0.0, sizeof(double) * 2);
            bool bDebug = false;
            if(i == 16961){
                bDebug = true;
            }
            double H[6][6] = {0}; ////  Hessian matrix
            // IterICGN2(ImRef, ImDef, gradxImgRef, gradyImgRef, localSubHom, localSub, p, pCoord, dispBias, maxIter, subset, halfSubset, sideW, length);
            calHessian(irow, icol, ImRef, ImDef, gradxImgRef, gradyImgRef, localSubHom, localSub,
                       p, pCoord, dispBias, maxIter, subset, halfSubset, sideW, length, H, bDebug);
            
            for (int n = 0; n < 36; n++)
            {
                int col = n % 6;
                int row = n / 6;
                *hessianMat.ptr<double>(n, i) = H[row][col];
//                if (i == 16961)
//                     printf("i: %d,irow: %d, icol: %d,  H[%d][%d]: %lf,disp[%d]: %d\n", i, irow, icol, row, col, H[row][col], i, disp[i]);
            }

            dispFloat[i] = disp[i] + dispBias[1];
            delete[] ImRef;
            delete[] ImDef;
            delete[] gradxImgRef;
            delete[] gradyImgRef;
            delete[] dispBias;
            // std::cout << "*********************************************" << std::endl;
        }
        else
        {
            dispFloat[i] = 0;
        }
    }

    cv::imwrite("./hessian_cpu.tif", hessianMat);
    return 1;
}
int CDispOptimizeICGN_CPU::calHessian(int row, int col, unsigned char *ImRef, unsigned char *ImDef, float *gradxImgRef, float *gradyImgRef,
                                      int *localSubHom, float *localSub, float p[], int pCoord[], double *dispBias, int maxIter,
                                      int subset, int halfSubset, int sideW, int length, double (*H)[6], bool bDebug)
{
    int sizeX = subset + 2 * sideW;
    int sizeY = subset + 2 * sideW;
    float M[6] = {1, 1.0f / subset, 1.0f / subset, 1, 1.0f / subset, 1.0f / subset};
    float MBT[4][4] = {{-0.166666666666667, 0.5, -0.5, 0.166666666666667},
                       {0.5, -1, 0, 0.666666666666667},
                       {-0.5, 0.5, 0.5, 0.166666666666667},
                       {0.166666666666667, 0, 0, 0}};

    double *nablafx = new double[subset * subset];
    double *nablafy = new double[subset * subset];

    double *Jacobian = new double[subset * subset * 6]; ////  Jacobian matrix
    int *fSubset = new int[subset * subset];
    // double H[6][6] = {0}; ////  Hessian matrix
    int n = 0;
    int sumfSubset = 0;
    for (int j = -halfSubset; j <= halfSubset; j++)
    {
        for (int k = -halfSubset; k <= halfSubset; k++)
        {
            fSubset[n] = int(ImRef[pCoord[1] * length + pCoord[0] + k * length + j]);
            sumfSubset += int(ImRef[pCoord[1] * length + pCoord[0] + k * length + j]);
            nablafx[n] = gradxImgRef[pCoord[1] * length + pCoord[0] + k * length + j];
            nablafy[n] = gradyImgRef[pCoord[1] * length + pCoord[0] + k * length + j];

            Jacobian[n] = nablafx[n];
            Jacobian[n + subset * subset] = nablafx[n] * localSub[n];//x
            Jacobian[n + 2 * subset * subset] = nablafx[n] * localSub[n + subset * subset];//y
            Jacobian[n + 3 * subset * subset] = nablafy[n];
            Jacobian[n + 4 * subset * subset] = nablafy[n] * localSub[n];
            Jacobian[n + 5 * subset * subset] = nablafy[n] * localSub[n + subset * subset];
            if (bDebug && (row + k) == 18)
            //if (bDebug)
            {
                // printf("j: %d, k: %d, n: %d, row: %d, col: %d,  gradxImgRef[%d]: %lf, gradyImgRef[%d]: %lf, ",
                // j, k, n, row + k,col + j,
                // pCoord[1] * length + pCoord[0] + k * length + j, nablafx[n],
                // pCoord[1] * length + pCoord[0] + k * length + j, nablafy[n]);
                // printf("Jacobian[0]: %lf, Jacobian[1]: %lf,Jacobian[2]: %lf,Jacobian[3]: %lf,Jacobian[4]: %lf,Jacobian[5]: %lf, ",
                //          Jacobian[n], Jacobian[n + subset * subset], Jacobian[n + 2 * subset * subset],
                //          Jacobian[n + 3 * subset * subset], Jacobian[n + 4 * subset * subset], Jacobian[n + 5 * subset * subset]);
                // printf("localSub[%d]: %lf, localSub[%d]: %lf\n", n, localSub[n], n + subset * subset, localSub[n + subset * subset]);
            }

            H[0][0] += Jacobian[n] * Jacobian[n];
            H[0][1] += Jacobian[n] * Jacobian[n + subset * subset];
            H[0][2] += Jacobian[n] * Jacobian[n + 2 * subset * subset];
            H[0][3] += Jacobian[n] * Jacobian[n + 3 * subset * subset];
            H[0][4] += Jacobian[n] * Jacobian[n + 4 * subset * subset];
            H[0][5] += Jacobian[n] * Jacobian[n + 5 * subset * subset];
            H[1][0] += Jacobian[n + subset * subset] * Jacobian[n];
            H[1][1] += Jacobian[n + subset * subset] * Jacobian[n + subset * subset];
            H[1][2] += Jacobian[n + subset * subset] * Jacobian[n + 2 * subset * subset];
            H[1][3] += Jacobian[n + subset * subset] * Jacobian[n + 3 * subset * subset];
            H[1][4] += Jacobian[n + subset * subset] * Jacobian[n + 4 * subset * subset];
            H[1][5] += Jacobian[n + subset * subset] * Jacobian[n + 5 * subset * subset];
            H[2][0] += Jacobian[n + 2 * subset * subset] * Jacobian[n];
            H[2][1] += Jacobian[n + 2 * subset * subset] * Jacobian[n + subset * subset];
            H[2][2] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 2 * subset * subset];
            H[2][3] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 3 * subset * subset];
            H[2][4] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 4 * subset * subset];
            H[2][5] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 5 * subset * subset];
            H[3][0] += Jacobian[n + 3 * subset * subset] * Jacobian[n];
            H[3][1] += Jacobian[n + 3 * subset * subset] * Jacobian[n + subset * subset];
            H[3][2] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 2 * subset * subset];
            H[3][3] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 3 * subset * subset];
            H[3][4] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 4 * subset * subset];
            H[3][5] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 5 * subset * subset];
            H[4][0] += Jacobian[n + 4 * subset * subset] * Jacobian[n];
            H[4][1] += Jacobian[n + 4 * subset * subset] * Jacobian[n + subset * subset];
            H[4][2] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 2 * subset * subset];
            H[4][3] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 3 * subset * subset];
            H[4][4] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 4 * subset * subset];
            H[4][5] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 5 * subset * subset];
            H[5][0] += Jacobian[n + 5 * subset * subset] * Jacobian[n];
            H[5][1] += Jacobian[n + 5 * subset * subset] * Jacobian[n + subset * subset];
            H[5][2] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 2 * subset * subset];
            H[5][3] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 3 * subset * subset];
            H[5][4] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 4 * subset * subset];
            H[5][5] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 5 * subset * subset];
            n++;
        }
    }

    for (int n = 0; n < 36; n++)
    {
        int icol = n % 6;
        int irow = n / 6;
        if(row == 13 && col == 321)
            printf("hessian[%d][%d]: %lf\n", irow, icol, H[irow][icol]);
    }

    return 0;
}
int CDispOptimizeICGN_CPU::IterICGN2(unsigned char *ImRef, unsigned char *ImDef, float *gradxImgRef, float *gradyImgRef,
                                     int *localSubHom, float *localSub, float p[], int pCoord[], double *dispBias, int maxIter,
                                     int subset, int halfSubset, int sideW, int length)
{
    int sizeX = subset + 2 * sideW;
    int sizeY = subset + 2 * sideW;
    float M[6] = {1, 1.0f / subset, 1.0f / subset, 1, 1.0f / subset, 1.0f / subset};
    float MBT[4][4] = {{-0.166666666666667, 0.5, -0.5, 0.166666666666667},
                       {0.5, -1, 0, 0.666666666666667},
                       {-0.5, 0.5, 0.5, 0.166666666666667},
                       {0.166666666666667, 0, 0, 0}};

    double *nablafx = new double[subset * subset];
    double *nablafy = new double[subset * subset];

    double *Jacobian = new double[subset * subset * 6]; ////  Jacobian matrix
    int *fSubset = new int[subset * subset];
    double H[6][6] = {0}; ////  Hessian matrix
    int n = 0;
    int sumfSubset = 0;
    for (int j = -halfSubset; j <= halfSubset; j++)
    {
        for (int k = -halfSubset; k <= halfSubset; k++)
        {
            fSubset[n] = int(ImRef[pCoord[1] * length + pCoord[0] + k * length + j]);
            sumfSubset += int(ImRef[pCoord[1] * length + pCoord[0] + k * length + j]);
            nablafx[n] = gradxImgRef[pCoord[1] * length + pCoord[0] + k * length + j];
            nablafy[n] = gradyImgRef[pCoord[1] * length + pCoord[0] + k * length + j];

            Jacobian[n] = nablafx[n];
            Jacobian[n + subset * subset] = nablafx[n] * localSub[n];
            Jacobian[n + 2 * subset * subset] = nablafx[n] * localSub[n + subset * subset];
            Jacobian[n + 3 * subset * subset] = nablafy[n];
            Jacobian[n + 4 * subset * subset] = nablafy[n] * localSub[n];
            Jacobian[n + 5 * subset * subset] = nablafy[n] * localSub[n + subset * subset];
            // cout << "n:          " << Jacobian[n] << "          " << Jacobian[n + subset * subset] << "           " << Jacobian[n + 2 * subset * subset] << "           " << Jacobian[n + 3 * subset * subset]  <<"           " << Jacobian[n + 4* subset * subset] << "           " << Jacobian[n + 5 * subset * subset] <<   endl;
            H[0][0] += Jacobian[n] * Jacobian[n];
            H[0][1] += Jacobian[n] * Jacobian[n + subset * subset];
            H[0][2] += Jacobian[n] * Jacobian[n + 2 * subset * subset];
            H[0][3] += Jacobian[n] * Jacobian[n + 3 * subset * subset];
            H[0][4] += Jacobian[n] * Jacobian[n + 4 * subset * subset];
            H[0][5] += Jacobian[n] * Jacobian[n + 5 * subset * subset];
            H[1][0] += Jacobian[n + subset * subset] * Jacobian[n];
            H[1][1] += Jacobian[n + subset * subset] * Jacobian[n + subset * subset];
            H[1][2] += Jacobian[n + subset * subset] * Jacobian[n + 2 * subset * subset];
            H[1][3] += Jacobian[n + subset * subset] * Jacobian[n + 3 * subset * subset];
            H[1][4] += Jacobian[n + subset * subset] * Jacobian[n + 4 * subset * subset];
            H[1][5] += Jacobian[n + subset * subset] * Jacobian[n + 5 * subset * subset];
            H[2][0] += Jacobian[n + 2 * subset * subset] * Jacobian[n];
            H[2][1] += Jacobian[n + 2 * subset * subset] * Jacobian[n + subset * subset];
            H[2][2] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 2 * subset * subset];
            H[2][3] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 3 * subset * subset];
            H[2][4] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 4 * subset * subset];
            H[2][5] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 5 * subset * subset];
            H[3][0] += Jacobian[n + 3 * subset * subset] * Jacobian[n];
            H[3][1] += Jacobian[n + 3 * subset * subset] * Jacobian[n + subset * subset];
            H[3][2] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 2 * subset * subset];
            H[3][3] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 3 * subset * subset];
            H[3][4] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 4 * subset * subset];
            H[3][5] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 5 * subset * subset];
            H[4][0] += Jacobian[n + 4 * subset * subset] * Jacobian[n];
            H[4][1] += Jacobian[n + 4 * subset * subset] * Jacobian[n + subset * subset];
            H[4][2] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 2 * subset * subset];
            H[4][3] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 3 * subset * subset];
            H[4][4] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 4 * subset * subset];
            H[4][5] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 5 * subset * subset];
            H[5][0] += Jacobian[n + 5 * subset * subset] * Jacobian[n];
            H[5][1] += Jacobian[n + 5 * subset * subset] * Jacobian[n + subset * subset];
            H[5][2] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 2 * subset * subset];
            H[5][3] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 3 * subset * subset];
            H[5][4] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 4 * subset * subset];
            H[5][5] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 5 * subset * subset];
            n++;
        }
    }

    double invH[6][6] = {0};
    GetMatrixInverse6(H, 6, invH);

    // cout << invH[0][0] << "    " << invH[0][1] << "    " << invH[0][2] << "    " << invH[0][3] << "    " << invH[0][4] << "    " << invH[0][5] << endl;
    // cout << invH[1][0] << "    " << invH[1][1] << "    " << invH[1][2] << "    " << invH[1][3] << "    " << invH[1][4] << "    " << invH[1][5] << endl;
    // cout << invH[2][0] << "    " << invH[2][1] << "    " << invH[2][2] << "    " << invH[2][3] << "    " << invH[2][4] << "    " << invH[2][5] << endl;
    // cout << invH[3][0] << "    " << invH[3][1] << "    " << invH[3][2] << "    " << invH[3][3] << "    " << invH[3][4] << "    " << invH[3][5] << endl;
    // cout << invH[4][0] << "    " << invH[4][1] << "    " << invH[4][2] << "    " << invH[4][3] << "    " << invH[4][4] << "    " << invH[4][5] << endl;
    // cout << invH[5][0] << "    " << invH[5][1] << "    " << invH[5][2] << "    " << invH[5][3] << "    " << invH[5][4] << "    " << invH[5][5] << endl;
    double *invHJacob = new double[6 * subset * subset];
    float *deltafVec = new float[subset * subset];
    float meanfSubset = sumfSubset / float(subset * subset);
    double sumRef2 = 0;
    for (int j = 0; j < subset * subset; j++)
    {
        invHJacob[j] = invH[0][0] * Jacobian[j] + invH[0][1] * Jacobian[j + subset * subset] + invH[0][2] * Jacobian[j + 2 * subset * subset] + invH[0][3] * Jacobian[j + 3 * subset * subset] + invH[0][4] * Jacobian[j + 4 * subset * subset] + invH[0][5] * Jacobian[j + 5 * subset * subset];
        invHJacob[j + subset * subset] = invH[1][0] * Jacobian[j] + invH[1][1] * Jacobian[j + subset * subset] + invH[1][2] * Jacobian[j + 2 * subset * subset] + invH[1][3] * Jacobian[j + 3 * subset * subset] + invH[1][4] * Jacobian[j + 4 * subset * subset] + invH[1][5] * Jacobian[j + 5 * subset * subset];
        invHJacob[j + 2 * subset * subset] = invH[2][0] * Jacobian[j] + invH[2][1] * Jacobian[j + subset * subset] + invH[2][2] * Jacobian[j + 2 * subset * subset] + invH[2][3] * Jacobian[j + 3 * subset * subset] + invH[2][4] * Jacobian[j + 4 * subset * subset] + invH[2][5] * Jacobian[j + 5 * subset * subset];
        invHJacob[j + 3 * subset * subset] = invH[3][0] * Jacobian[j] + invH[3][1] * Jacobian[j + subset * subset] + invH[3][2] * Jacobian[j + 2 * subset * subset] + invH[3][3] * Jacobian[j + 3 * subset * subset] + invH[3][4] * Jacobian[j + 4 * subset * subset] + invH[3][5] * Jacobian[j + 5 * subset * subset];
        invHJacob[j + 4 * subset * subset] = invH[4][0] * Jacobian[j] + invH[4][1] * Jacobian[j + subset * subset] + invH[4][2] * Jacobian[j + 2 * subset * subset] + invH[4][3] * Jacobian[j + 3 * subset * subset] + invH[4][4] * Jacobian[j + 4 * subset * subset] + invH[4][5] * Jacobian[j + 5 * subset * subset];
        invHJacob[j + 5 * subset * subset] = invH[5][0] * Jacobian[j] + invH[5][1] * Jacobian[j + subset * subset] + invH[5][2] * Jacobian[j + 2 * subset * subset] + invH[5][3] * Jacobian[j + 3 * subset * subset] + invH[5][4] * Jacobian[j + 4 * subset * subset] + invH[5][5] * Jacobian[j + 5 * subset * subset];
        // cout << "j:          " << j << "           " << invHJacob[j] << "          " << invHJacob[j + subset * subset] << "           " << invHJacob[j + 2 * subset * subset] << "           " << invHJacob[j + 3 * subset * subset] << "           " << invHJacob[j + 4 * subset * subset] << "           " << invHJacob[j + 5 * subset * subset] << endl;
        ////////////////////////////////////////////////////////////////////////////////////////
        deltafVec[j] = float(fSubset[j] - meanfSubset);
        // cout << "j:         " << j << "          " << deltafVec[j] << endl;
        sumRef2 += deltafVec[j] * deltafVec[j];
    }
    double deltaf = sqrt(sumRef2);

    double warP[3][3] = {{1 + p[1], p[2], p[0]}, {p[4], 1 + p[5], p[3]}, {0, 0, 1}};
    double thre = 1;
    int Iter = 0;
    double Czncc = 0;
    double *gIntep = new double[3 * subset * subset];
    double *PcoordInt = new double[3 * subset * subset];
    while (thre > 1e-3 && Iter < maxIter || Iter == 0)
    {
        double minPcoordInt1 = subset;
        double minPcoordInt2 = subset;
        double maxPcoordInt1 = -subset;
        double maxPcoordInt2 = -subset;
        for (int j = 0; j < subset * subset; j++)
        {
            gIntep[j] = warP[0][0] * localSubHom[j] + warP[0][1] * localSubHom[j + subset * subset] + warP[0][2] * localSubHom[j + 2 * subset * subset];
            gIntep[j + subset * subset] = warP[1][0] * localSubHom[j] + warP[1][1] * localSubHom[j + subset * subset] + warP[1][2] * localSubHom[j + 2 * subset * subset];
            gIntep[j + 2 * subset * subset] = warP[2][0] * localSubHom[j] + warP[2][1] * localSubHom[j + subset * subset] + warP[2][2] * localSubHom[j + 2 * subset * subset];
            // cout << gIntep[j] << "              " << gIntep[j + subset * subset] << "             " << gIntep[j + 2 * subset * subset] << endl;
            PcoordInt[j] = pCoord[0] + gIntep[j];
            PcoordInt[j + subset * subset] = pCoord[1] + gIntep[j + subset * subset];
            PcoordInt[j + 2 * subset * subset] = pCoord[2] + gIntep[j + 2 * subset * subset] - 1;
            if (PcoordInt[j] < minPcoordInt1)
            {
                minPcoordInt1 = PcoordInt[j];
            }
            if (PcoordInt[j + subset * subset] < minPcoordInt2)
            {
                minPcoordInt2 = PcoordInt[j + subset * subset];
            }
            if (PcoordInt[j] > maxPcoordInt1)
            {
                maxPcoordInt1 = PcoordInt[j];
            }
            if (PcoordInt[j + subset * subset] > maxPcoordInt2)
            {
                maxPcoordInt2 = PcoordInt[j + subset * subset];
            }
        }

        if (minPcoordInt1 > 2 && minPcoordInt2 > 2 && maxPcoordInt1 < sizeX - 2 && maxPcoordInt2 < sizeY - 2)
        {
            int *xInt = new int[3 * subset * subset];
            double *deltaX = new double[3 * subset * subset];
            double *deltaMatX = new double[4 * subset * subset];
            double *deltaMatY = new double[4 * subset * subset];
            int *Indx = new int[16];
            int *D_all = new int[16];
            double *defIntp = new double[16];
            double *defIntp2 = new double[subset * subset];
            double sumdefIntp = 0;

            for (int j = 0; j < subset * subset; j++)
            {
                xInt[j] = floor(PcoordInt[j]);
                xInt[j + subset * subset] = floor(PcoordInt[j + subset * subset]);
                xInt[j + 2 * subset * subset] = floor(PcoordInt[j + 2 * subset * subset]);

                // cout << "j:          " << j << "          " << xInt[j] << "          " << xInt[j + subset * subset] << "          " << xInt[j + 2 * subset * subset] << endl;
                deltaX[j] = PcoordInt[j] - xInt[j];
                deltaX[j + subset * subset] = PcoordInt[j + subset * subset] - xInt[j + subset * subset];
                deltaX[j + 2 * subset * subset] = PcoordInt[j + 2 * subset * subset] - xInt[j + 2 * subset * subset];
                deltaMatX[j] = MBT[0][0] * deltaX[j] * deltaX[j] * deltaX[j] + MBT[0][1] * deltaX[j] * deltaX[j] + MBT[0][2] * deltaX[j] + MBT[0][3];
                deltaMatX[j + subset * subset] = MBT[1][0] * deltaX[j] * deltaX[j] * deltaX[j] + MBT[1][1] * deltaX[j] * deltaX[j] + MBT[1][2] * deltaX[j] + MBT[1][3];
                deltaMatX[j + 2 * subset * subset] = MBT[2][0] * deltaX[j] * deltaX[j] * deltaX[j] + MBT[2][1] * deltaX[j] * deltaX[j] + MBT[2][2] * deltaX[j] + MBT[2][3];
                deltaMatX[j + 3 * subset * subset] = MBT[3][0] * deltaX[j] * deltaX[j] * deltaX[j] + MBT[3][1] * deltaX[j] * deltaX[j] + MBT[3][2] * deltaX[j] + MBT[3][3];

                // cout << "j:          " << j << "          " << deltaMatX[j] << "              " << deltaMatX[j + subset * subset] << "             " << deltaMatX[j + 2 * subset * subset] << "              " << deltaMatX[j + 3 * subset * subset] << "             " << endl;

                deltaMatY[j] = MBT[0][0] * deltaX[j + subset * subset] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[0][1] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[0][2] * deltaX[j + subset * subset] + MBT[0][3];
                deltaMatY[j + subset * subset] = MBT[1][0] * deltaX[j + subset * subset] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[1][1] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[1][2] * deltaX[j + subset * subset] + MBT[1][3];
                deltaMatY[j + 2 * subset * subset] = MBT[2][0] * deltaX[j + subset * subset] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[2][1] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[2][2] * deltaX[j + subset * subset] + MBT[2][3];
                deltaMatY[j + 3 * subset * subset] = MBT[3][0] * deltaX[j + subset * subset] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[3][1] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[3][2] * deltaX[j + subset * subset] + MBT[3][3];
                // cout << "j:          " << j << "          " << deltaMatY[j] << "              " << deltaMatY[j + subset * subset] << "             " << deltaMatY[j + 2 * subset * subset] << "              " << deltaMatY[j + 3 * subset * subset] << "             " << endl;
                Indx[0] = (xInt[j + subset * subset] - 1) * length + xInt[j];
                Indx[1] = (xInt[j + subset * subset]) * length + xInt[j];
                Indx[2] = (xInt[j + subset * subset] + 1) * length + xInt[j];
                Indx[3] = (xInt[j + subset * subset] + 2) * length + xInt[j];
                Indx[4] = (xInt[j + subset * subset] - 1) * length + xInt[j] + 1;
                Indx[5] = (xInt[j + subset * subset]) * length + xInt[j] + 1;
                Indx[6] = (xInt[j + subset * subset] + 1) * length + xInt[j] + 1;
                Indx[7] = (xInt[j + subset * subset] + 2) * length + xInt[j] + 1;
                Indx[8] = (xInt[j + subset * subset] - 1) * length + xInt[j] + 2;
                Indx[9] = (xInt[j + subset * subset]) * length + xInt[j] + 2;
                Indx[10] = (xInt[j + subset * subset] + 1) * length + xInt[j] + 2;
                Indx[11] = (xInt[j + subset * subset] + 2) * length + xInt[j] + 2;
                Indx[12] = (xInt[j + subset * subset] - 1) * length + xInt[j] + 3;
                Indx[13] = (xInt[j + subset * subset]) * length + xInt[j] + 3;
                Indx[14] = (xInt[j + subset * subset] + 1) * length + xInt[j] + 3;
                Indx[15] = (xInt[j + subset * subset] + 2) * length + xInt[j] + 3;
                // for (int ic = 0; ic < 16; ic++)
                //{
                //	cout << "j:          " << j << "          " << "ic:          " << ic << "          " << Indx[ic] << endl;
                // }
                // cout << " ***********************************************************" << endl;
                // cout << "j:          " << j << "          " << deltaMatY[j] << "              " << deltaMatY[j + subset * subset] << "             " << deltaMatY[j + 2 * subset * subset] << "              " << deltaMatY[j + 3 * subset * subset] << "             " << endl;
                D_all[0] = ImDef[(Indx[0] % length - 1) * length + Indx[0] / length];
                D_all[1] = ImDef[(Indx[1] % length - 1) * length + Indx[1] / length];
                D_all[2] = ImDef[(Indx[2] % length - 1) * length + Indx[2] / length];
                D_all[3] = ImDef[(Indx[3] % length - 1) * length + Indx[3] / length];
                D_all[4] = ImDef[(Indx[4] % length - 1) * length + Indx[4] / length];
                D_all[5] = ImDef[(Indx[5] % length - 1) * length + Indx[5] / length];
                D_all[6] = ImDef[(Indx[6] % length - 1) * length + Indx[6] / length];
                D_all[7] = ImDef[(Indx[7] % length - 1) * length + Indx[7] / length];
                D_all[8] = ImDef[(Indx[8] % length - 1) * length + Indx[8] / length];
                D_all[9] = ImDef[(Indx[9] % length - 1) * length + Indx[9] / length];
                D_all[10] = ImDef[(Indx[10] % length - 1) * length + Indx[10] / length];
                D_all[11] = ImDef[(Indx[11] % length - 1) * length + Indx[11] / length];
                D_all[12] = ImDef[(Indx[12] % length - 1) * length + Indx[12] / length];
                D_all[13] = ImDef[(Indx[13] % length - 1) * length + Indx[13] / length];
                D_all[14] = ImDef[(Indx[14] % length - 1) * length + Indx[14] / length];
                D_all[15] = ImDef[(Indx[15] % length - 1) * length + Indx[15] / length];

                defIntp[0] = deltaMatY[j] * D_all[0] * deltaMatX[j];
                defIntp[1] = deltaMatY[j + subset * subset] * D_all[1] * deltaMatX[j];
                defIntp[2] = deltaMatY[j + 2 * subset * subset] * D_all[2] * deltaMatX[j];
                defIntp[3] = deltaMatY[j + 3 * subset * subset] * D_all[3] * deltaMatX[j];
                defIntp[4] = deltaMatY[j] * D_all[4] * deltaMatX[j + subset * subset];
                defIntp[5] = deltaMatY[j + subset * subset] * D_all[5] * deltaMatX[j + subset * subset];
                defIntp[6] = deltaMatY[j + 2 * subset * subset] * D_all[6] * deltaMatX[j + subset * subset];
                defIntp[7] = deltaMatY[j + 3 * subset * subset] * D_all[7] * deltaMatX[j + subset * subset];
                defIntp[8] = deltaMatY[j] * D_all[8] * deltaMatX[j + 2 * subset * subset];
                defIntp[9] = deltaMatY[j + subset * subset] * D_all[9] * deltaMatX[j + 2 * subset * subset];
                defIntp[10] = deltaMatY[j + 2 * subset * subset] * D_all[10] * deltaMatX[j + 2 * subset * subset];
                defIntp[11] = deltaMatY[j + 3 * subset * subset] * D_all[11] * deltaMatX[j + 2 * subset * subset];
                defIntp[12] = deltaMatY[j] * D_all[12] * deltaMatX[j + 3 * subset * subset];
                defIntp[13] = deltaMatY[j + subset * subset] * D_all[13] * deltaMatX[j + 3 * subset * subset];
                defIntp[14] = deltaMatY[j + 2 * subset * subset] * D_all[14] * deltaMatX[j + 3 * subset * subset];
                defIntp[15] = deltaMatY[j + 3 * subset * subset] * D_all[15] * deltaMatX[j + 3 * subset * subset];
                // for (int ic = 0; ic < 16; ic++)
                //{
                //	cout << "j:          " << j << "          " << "ic:          " << ic << "          " << defIntp[ic] << endl;
                // }
                // cout << " ***********************************************************" << endl;
                defIntp2[j] = defIntp[0] + defIntp[1] + defIntp[2] + defIntp[3] + defIntp[4] + defIntp[5] + defIntp[6] + defIntp[7] + defIntp[8] + defIntp[9] + defIntp[10] + defIntp[11] + defIntp[12] + defIntp[13] + defIntp[14] + defIntp[15];
                sumdefIntp += defIntp2[j];
            }

            float meandefIntp = sumdefIntp / float(subset * subset);

            double *deltagVec = new double[subset * subset];
            double sumdeltagVec = 0;
            for (int j = 0; j < subset * subset; j++)
            {
                deltagVec[j] = defIntp2[j] - meandefIntp;
                sumdeltagVec += (deltagVec[j] * deltagVec[j]);
            }
            double deltag = sqrt(sumdeltagVec);
            double *delta = new double[subset * subset];
            double deltap[6] = {0};
            for (int j = 0; j < subset * subset; j++)
            {
                delta[j] = deltafVec[j] - deltaf / deltag * deltagVec[j];
                deltap[0] += -invHJacob[j] * delta[j];
                deltap[1] += -invHJacob[j + subset * subset] * delta[j];
                deltap[2] += -invHJacob[j + 2 * subset * subset] * delta[j];
                deltap[3] += -invHJacob[j + 3 * subset * subset] * delta[j];
                deltap[4] += -invHJacob[j + 4 * subset * subset] * delta[j];
                deltap[5] += -invHJacob[j + 5 * subset * subset] * delta[j];
            }
            deltap[0] = M[0] * deltap[0];
            deltap[1] = M[1] * deltap[1];
            deltap[2] = M[2] * deltap[2];
            deltap[3] = M[3] * deltap[3];
            deltap[4] = M[4] * deltap[4];
            deltap[5] = M[5] * deltap[5];

            double warpdelta[3][3] = {{1 + deltap[1], deltap[2], deltap[0]}, {deltap[4], 1 + deltap[5], deltap[3]}, {0, 0, 1}};
            double invwarpdelta[3][3] = {0};
            GetMatrixInverse(warpdelta, 3, invwarpdelta);

            double warP2[3][3] = {0};
            warP2[0][0] = warP[0][0] * invwarpdelta[0][0] + warP[0][1] * invwarpdelta[1][0] + warP[0][2] * invwarpdelta[2][0];
            warP2[0][1] = warP[0][0] * invwarpdelta[0][1] + warP[0][1] * invwarpdelta[1][1] + warP[0][2] * invwarpdelta[2][1];
            warP2[0][2] = warP[0][0] * invwarpdelta[0][2] + warP[0][1] * invwarpdelta[1][2] + warP[0][2] * invwarpdelta[2][2];
            warP2[1][0] = warP[1][0] * invwarpdelta[0][0] + warP[1][1] * invwarpdelta[1][0] + warP[1][2] * invwarpdelta[2][0];
            warP2[1][1] = warP[1][0] * invwarpdelta[0][1] + warP[1][1] * invwarpdelta[1][1] + warP[1][2] * invwarpdelta[2][1];
            warP2[1][2] = warP[1][0] * invwarpdelta[0][2] + warP[1][1] * invwarpdelta[1][2] + warP[1][2] * invwarpdelta[2][2];
            warP2[2][0] = warP[2][0] * invwarpdelta[0][0] + warP[2][1] * invwarpdelta[1][0] + warP[2][2] * invwarpdelta[2][0];
            warP2[2][1] = warP[2][0] * invwarpdelta[0][1] + warP[2][1] * invwarpdelta[1][1] + warP[2][2] * invwarpdelta[2][1];
            warP2[2][2] = warP[2][0] * invwarpdelta[0][2] + warP[2][1] * invwarpdelta[1][2] + warP[2][2] * invwarpdelta[2][2];

            warP[0][0] = warP2[0][0], warP[0][1] = warP2[0][1], warP[0][2] = warP2[0][2];
            warP[1][0] = warP2[1][0], warP[1][1] = warP2[1][1], warP[1][2] = warP2[1][2];
            warP[2][0] = warP2[2][0], warP[2][1] = warP2[2][1], warP[2][2] = warP2[2][2];

            // cout << warP[0][0] << "        " << warP[0][1] << "        " << warP[0][2] << endl;
            // cout << warP[1][0] << "        " << warP[1][1] << "        " << warP[1][2] << endl;
            // cout << warP[2][0] << "        " << warP[2][1] << "        " << warP[2][2] << endl;

            thre = sqrt(deltap[0] * deltap[0] + deltap[3] * deltap[3]);
            // cout << "thre:              " << thre << endl;
            Iter++;
            p[0] = warP[0][2];
            p[1] = warP[0][0] - 1;
            p[2] = warP[0][1];
            p[3] = warP[1][2];
            p[4] = warP[1][0];
            p[5] = warP[1][1] - 1;
            // cout << p[0] << "        " << p[1] << "        " << p[2] << "        " << p[3] << "        " << p[4] << "        " << p[5] <<  endl;
            dispBias[0] = p[0];
            dispBias[1] = p[3];
            double Cznssd = 0;
            for (int j = 0; j < subset * subset; j++)
            {
                double deltafg = (deltafVec[j] / deltaf - deltagVec[j] / deltag);
                Cznssd += deltafg * deltafg;
            }
            Czncc = 1 - 0.5 * Cznssd;
            delete[] deltagVec;
            delete[] defIntp2;
            delete[] defIntp;
            delete[] D_all;
            delete[] Indx;
            delete[] deltaMatY;
            delete[] deltaMatX;
            delete[] deltaX;
            delete[] delta;
            delete[] xInt;
        }
        else
        {
            Iter = Iter + 1;
            p[0] = 0, p[1] = 0, p[2] = 0, p[3] = 0, p[4] = 0, p[5] = 0;
            dispBias[0] = 0;
            dispBias[1] = 0;
            Czncc = -1;
            break;
        }
    }
    delete[] gIntep;
    delete[] PcoordInt;
    delete[] invHJacob;
    delete[] deltafVec;
    delete[] fSubset;
    delete[] Jacobian;
    // delete[] nablafxy;
    // delete[] nablafyy;
    // delete[] nablafxx;
    delete[] nablafy;
    delete[] nablafx;
    return 1;
}
// 得到给定矩阵src的逆矩阵保存到des中。
int CDispOptimizeICGN_CPU::GetMatrixInverse6(double src[6][6], int n, double des[6][6])
{
    double flag = getA6(src, n);
    double t[6][6];
    if (flag == 0)
    {
        return 0;
    }
    else
    {
        getAStart6(src, n, t);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                des[i][j] = t[i][j] / flag;
            }
        }
    }
    return 1;
}

// 得到给定矩阵src的逆矩阵保存到des中。
int CDispOptimizeICGN_CPU::GetMatrixInverse(double src[3][3], int n, double des[3][3])
{
    double flag = getA(src, n);
    double t[3][3];
    if (flag == 0)
    {
        return 0;
    }
    else
    {
        getAStart(src, n, t);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                des[i][j] = t[i][j] / flag;
            }
        }
    }
    return 1;
}

// 按第一行展开计算|A|
double CDispOptimizeICGN_CPU::getA(double arcs[3][3], int n)
{
    if (n == 1)
    {
        return arcs[0][0];
    }
    double ans = 0;
    double temp[3][3] = {0.0};
    int i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n - 1; j++)
        {
            for (k = 0; k < n - 1; k++)
            {
                temp[j][k] = arcs[j + 1][(k >= i) ? k + 1 : k];
            }
        }
        double t = getA(temp, n - 1);
        if (i % 2 == 0)
        {
            ans += arcs[0][i] * t;
        }
        else
        {
            ans -= arcs[0][i] * t;
        }
    }
    return ans;
}

// 计算每一行每一列的每个元素所对应的余子式，组成A*
int CDispOptimizeICGN_CPU::getAStart6(double arcs[6][6], int n, double ans[6][6])
{
    if (n == 1)
    {
        ans[0][0] = 1;
        return 0;
    }
    int i, j, k, t;
    double temp[6][6];
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n - 1; k++)
            {
                for (t = 0; t < n - 1; t++)
                {
                    temp[k][t] = arcs[k >= i ? k + 1 : k][t >= j ? t + 1 : t];
                }
            }

            ans[j][i] = getA6(temp, n - 1);
            if ((i + j) % 2 == 1)
            {
                ans[j][i] = -ans[j][i];
            }
        }
    }
    return 1;
}

// 计算每一行每一列的每个元素所对应的余子式，组成A*
int CDispOptimizeICGN_CPU::getAStart(double arcs[3][3], int n, double ans[3][3])
{
    if (n == 1)
    {
        ans[0][0] = 1;
        return 0;
    }
    int i, j, k, t;
    double temp[3][3];
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n - 1; k++)
            {
                for (t = 0; t < n - 1; t++)
                {
                    temp[k][t] = arcs[k >= i ? k + 1 : k][t >= j ? t + 1 : t];
                }
            }

            ans[j][i] = getA(temp, n - 1);
            if ((i + j) % 2 == 1)
            {
                ans[j][i] = -ans[j][i];
            }
        }
    }
    return 1;
}

double CDispOptimizeICGN_CPU::getA6(double arcs[6][6], int n)
{
    if (n == 1)
    {
        return arcs[0][0];
    }
    double ans = 0;
    double temp[6][6] = {0.0};
    int i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n - 1; j++)
        {
            for (k = 0; k < n - 1; k++)
            {
                temp[j][k] = arcs[j + 1][(k >= i) ? k + 1 : k];
            }
        }
        double t = getA6(temp, n - 1);
        if (i % 2 == 0)
        {
            ans += arcs[0][i] * t;
        }
        else
        {
            ans -= arcs[0][i] * t;
        }
    }
    return ans;
}

int CDispOptimizeICGN_CPU::ICGNalgo_new(cv::Mat &_l_image, cv::Mat &_r_image, cv::Mat &_src_disp, cv::Mat &gradImage_x,
                                        cv::Mat &gradImage_y, cv::Mat &localSubHom, cv::Mat &localSub,
                                        cv::Mat &opt_disp, int subset, int sideW, int maxIter)
{

    int halfSubset = subset / 2;
    int halfWinSize = halfSubset + sideW; // 7+5;
    int pCoord[3] = {halfWinSize, halfWinSize, 1};
    int length = 2 * halfWinSize + 1; // 25=2*12 + 1;
    int sizeX = subset + 2 * sideW;   // 15+2*5=25;
    int sizeY = subset + 2 * sideW;
    int width = _l_image.cols;
    int height = _l_image.rows;
    for (int iRow = 0; iRow < _l_image.rows; ++iRow)
    {
        for (int iCol = 0; iCol < _l_image.cols; ++iCol)
        {
            uchar disp_value = *_src_disp.ptr<uchar>(iRow, iCol);
            bool bRet = false;
            bRet = disp_value > 0;
            bRet &= (iRow - halfWinSize > 0);
            bRet &= (iRow + halfWinSize < height);
            bRet &= (iCol - halfWinSize > 0);
            bRet &= (iCol + halfWinSize < width);
            bRet &= (iCol - disp_value - halfWinSize > 0);
            bRet &= (iCol - disp_value + halfWinSize < width);
            if (bRet)
            {
                cv::Mat l_sub_region = _l_image(cv::Rect(iCol - halfWinSize, iRow - halfWinSize, sizeX, sizeY));
                cv::Mat r_sub_region = _r_image(cv::Rect(iCol - halfWinSize - disp_value, iRow - halfWinSize, sizeX, sizeY));
                cv::Mat gradx_sub_region = gradImage_x(cv::Rect(iCol - halfWinSize, iRow - halfWinSize, sizeX, sizeY));
                cv::Mat grady_sub_region = gradImage_y(cv::Rect(iCol - halfWinSize, iRow - halfWinSize, sizeX, sizeY));

                float p[6] = {0.4686, 0, 0, -0.2116, 0, 0};
                double *dispBias = new double[2];
                memset(dispBias, 0.0, sizeof(double) * 2);
                CalOptDisp(l_sub_region, r_sub_region, gradx_sub_region, grady_sub_region,
                           localSubHom, localSub, p, pCoord, dispBias, maxIter, subset, halfSubset, sideW, length);
                *opt_disp.ptr<double>(iRow, iCol) = disp_value + dispBias[1];
            }
        }
    }
    return 0;
}

int CDispOptimizeICGN_CPU::CalOptDisp(cv::Mat &ImRef, cv::Mat &ImDef, cv::Mat &gradxImgRef, cv::Mat &gradyImgRef,
                                      cv::Mat &localSubHom, cv::Mat &localSub, float *p, int *pCoord, double *dispBias,
                                      int maxIter, int subset, int halfSubset, int sideW, int length)
{
    //    int sizeX = subset + 2 * sideW;
    //    int sizeY = subset + 2 * sideW;
    //    float M[6] = { 1, 1.0f / subset,  1.0f / subset, 1, 1.0f / subset, 1.0f / subset };
    //    float MBT[4][4] = { {-0.166666666666667, 0.5, -0.5, 0.166666666666667},
    //                        {0.5, -1, 0, 0.666666666666667},
    //                        {-0.5, 0.5, 0.5, 0.166666666666667},
    //                        {0.166666666666667, 0, 0, 0} };

    //    //double* nablafx = new double[subset * subset];
    //    //double* nablafy = new double[subset * subset];

    //    cv::Mat nablafx = cv::Mat(subset, subset, CV_64FC1);
    //    cv::Mat nablafy = cv::Mat(subset, subset, CV_64FC1);
    //    cv::Mat Jacobian = cv::Mat(subset * subset, 6, CV_64FC1);
    //    cv::Mat fSubset = cv::Mat(subset,  subset, CV_32SC1);

    //    cv::Mat hessianMat = cv::Mat(6, 6, CV_64FC1,0);
    //    //double H[6][6] = { 0 };  ////  Hessian matrix
    //    int n = 0;
    //    int sumfSubset = 0;
    //    for (int j = -halfSubset; j <= halfSubset; j++)
    //    {
    //        for (int k = -halfSubset; k <= halfSubset; k++)
    //        {
    //            fSubset[n] = int(ImRef[pCoord[1] * length + pCoord[0] + k * length + j]);
    //            sumfSubset += int( ImRef[pCoord[1] * length + pCoord[0] + k * length + j]);
    //            nablafx[n] = gradxImgRef[pCoord[1] * length + pCoord[0] + k * length + j];
    //            nablafy[n] = gradyImgRef[pCoord[1] * length + pCoord[0] + k * length + j];

    //            Jacobian[n] = nablafx[n];
    //            Jacobian[n + subset * subset] = nablafx[n] * localSub[n];
    //            Jacobian[n + 2 * subset * subset] = nablafx[n] * localSub[n + subset * subset];
    //            Jacobian[n + 3 * subset * subset] = nablafy[n];
    //            Jacobian[n + 4 * subset * subset] = nablafy[n] * localSub[n];
    //            Jacobian[n + 5 * subset * subset] = nablafy[n] * localSub[n + subset * subset];
    //            //cout << "n:          " << Jacobian[n] << "          " << Jacobian[n + subset * subset] << "           " << Jacobian[n + 2 * subset * subset] << "           " << Jacobian[n + 3 * subset * subset]  <<"           " << Jacobian[n + 4* subset * subset] << "           " << Jacobian[n + 5 * subset * subset] <<   endl;
    //            H[0][0] += Jacobian[n] * Jacobian[n];
    //            H[0][1] += Jacobian[n] * Jacobian[n + subset * subset];
    //            H[0][2] += Jacobian[n] * Jacobian[n + 2 * subset * subset];
    //            H[0][3] += Jacobian[n] * Jacobian[n + 3 * subset * subset];
    //            H[0][4] += Jacobian[n] * Jacobian[n + 4 * subset * subset];
    //            H[0][5] += Jacobian[n] * Jacobian[n + 5 * subset * subset];
    //            H[1][0] += Jacobian[n + subset * subset] * Jacobian[n];
    //            H[1][1] += Jacobian[n + subset * subset] * Jacobian[n + subset * subset];
    //            H[1][2] += Jacobian[n + subset * subset] * Jacobian[n + 2 * subset * subset];
    //            H[1][3] += Jacobian[n + subset * subset] * Jacobian[n + 3 * subset * subset];
    //            H[1][4] += Jacobian[n + subset * subset] * Jacobian[n + 4 * subset * subset];
    //            H[1][5] += Jacobian[n + subset * subset] * Jacobian[n + 5 * subset * subset];
    //            H[2][0] += Jacobian[n + 2 * subset * subset] * Jacobian[n];
    //            H[2][1] += Jacobian[n + 2 * subset * subset] * Jacobian[n + subset * subset];
    //            H[2][2] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 2 * subset * subset];
    //            H[2][3] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 3 * subset * subset];
    //            H[2][4] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 4 * subset * subset];
    //            H[2][5] += Jacobian[n + 2 * subset * subset] * Jacobian[n + 5 * subset * subset];
    //            H[3][0] += Jacobian[n + 3 * subset * subset] * Jacobian[n];
    //            H[3][1] += Jacobian[n + 3 * subset * subset] * Jacobian[n + subset * subset];
    //            H[3][2] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 2 * subset * subset];
    //            H[3][3] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 3 * subset * subset];
    //            H[3][4] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 4 * subset * subset];
    //            H[3][5] += Jacobian[n + 3 * subset * subset] * Jacobian[n + 5 * subset * subset];
    //            H[4][0] += Jacobian[n + 4 * subset * subset] * Jacobian[n];
    //            H[4][1] += Jacobian[n + 4 * subset * subset] * Jacobian[n + subset * subset];
    //            H[4][2] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 2 * subset * subset];
    //            H[4][3] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 3 * subset * subset];
    //            H[4][4] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 4 * subset * subset];
    //            H[4][5] += Jacobian[n + 4 * subset * subset] * Jacobian[n + 5 * subset * subset];
    //            H[5][0] += Jacobian[n + 5 * subset * subset] * Jacobian[n];
    //            H[5][1] += Jacobian[n + 5 * subset * subset] * Jacobian[n + subset * subset];
    //            H[5][2] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 2 * subset * subset];
    //            H[5][3] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 3 * subset * subset];
    //            H[5][4] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 4 * subset * subset];
    //            H[5][5] += Jacobian[n + 5 * subset * subset] * Jacobian[n + 5 * subset * subset];
    //            n++;
    //        }
    //    }

    //    double invH[6][6] = { 0 };
    //    GetMatrixInverse6(H, 6, invH);

    //    //cout << invH[0][0] << "    " << invH[0][1] << "    " << invH[0][2] << "    " << invH[0][3] << "    " << invH[0][4] << "    " << invH[0][5] << endl;
    //    //cout << invH[1][0] << "    " << invH[1][1] << "    " << invH[1][2] << "    " << invH[1][3] << "    " << invH[1][4] << "    " << invH[1][5] << endl;
    //    //cout << invH[2][0] << "    " << invH[2][1] << "    " << invH[2][2] << "    " << invH[2][3] << "    " << invH[2][4] << "    " << invH[2][5] << endl;
    //    //cout << invH[3][0] << "    " << invH[3][1] << "    " << invH[3][2] << "    " << invH[3][3] << "    " << invH[3][4] << "    " << invH[3][5] << endl;
    //    //cout << invH[4][0] << "    " << invH[4][1] << "    " << invH[4][2] << "    " << invH[4][3] << "    " << invH[4][4] << "    " << invH[4][5] << endl;
    //    //cout << invH[5][0] << "    " << invH[5][1] << "    " << invH[5][2] << "    " << invH[5][3] << "    " << invH[5][4] << "    " << invH[5][5] << endl;
    //    double* invHJacob = new double[6 * subset * subset];
    //    float* deltafVec = new float[subset * subset];
    //    float meanfSubset = sumfSubset / float(subset * subset);
    //    double sumRef2 = 0;
    //    for (int j = 0; j < subset * subset; j++)
    //    {
    //        invHJacob[j] = invH[0][0] * Jacobian[j] + invH[0][1] * Jacobian[j + subset * subset] + invH[0][2] * Jacobian[j + 2 * subset * subset] + invH[0][3] * Jacobian[j + 3 * subset * subset] + invH[0][4] * Jacobian[j + 4 * subset * subset] + invH[0][5] * Jacobian[j + 5 * subset * subset];
    //        invHJacob[j + subset * subset] = invH[1][0] * Jacobian[j] + invH[1][1] * Jacobian[j + subset * subset] + invH[1][2] * Jacobian[j + 2 * subset * subset] + invH[1][3] * Jacobian[j + 3 * subset * subset] + invH[1][4] * Jacobian[j + 4 * subset * subset] + invH[1][5] * Jacobian[j + 5 * subset * subset];
    //        invHJacob[j + 2 * subset * subset] = invH[2][0] * Jacobian[j] + invH[2][1] * Jacobian[j + subset * subset] + invH[2][2] * Jacobian[j + 2 * subset * subset] + invH[2][3] * Jacobian[j + 3 * subset * subset] + invH[2][4] * Jacobian[j + 4 * subset * subset] + invH[2][5] * Jacobian[j + 5 * subset * subset];
    //        invHJacob[j + 3 * subset * subset] = invH[3][0] * Jacobian[j] + invH[3][1] * Jacobian[j + subset * subset] + invH[3][2] * Jacobian[j + 2 * subset * subset] + invH[3][3] * Jacobian[j + 3 * subset * subset] + invH[3][4] * Jacobian[j + 4 * subset * subset] + invH[3][5] * Jacobian[j + 5 * subset * subset];
    //        invHJacob[j + 4 * subset * subset] = invH[4][0] * Jacobian[j] + invH[4][1] * Jacobian[j + subset * subset] + invH[4][2] * Jacobian[j + 2 * subset * subset] + invH[4][3] * Jacobian[j + 3 * subset * subset] + invH[4][4] * Jacobian[j + 4 * subset * subset] + invH[4][5] * Jacobian[j + 5 * subset * subset];
    //        invHJacob[j + 5 * subset * subset] = invH[5][0] * Jacobian[j] + invH[5][1] * Jacobian[j + subset * subset] + invH[5][2] * Jacobian[j + 2 * subset * subset] + invH[5][3] * Jacobian[j + 3 * subset * subset] + invH[5][4] * Jacobian[j + 4 * subset * subset] + invH[5][5] * Jacobian[j + 5 * subset * subset];
    //        //cout << "j:          " << j << "           " << invHJacob[j] << "          " << invHJacob[j + subset * subset] << "           " << invHJacob[j + 2 * subset * subset] << "           " << invHJacob[j + 3 * subset * subset] << "           " << invHJacob[j + 4 * subset * subset] << "           " << invHJacob[j + 5 * subset * subset] << endl;
    //        ////////////////////////////////////////////////////////////////////////////////////////
    //        deltafVec[j] = float(fSubset[j] - meanfSubset);
    //        //cout << "j:         " << j << "          " << deltafVec[j] << endl;
    //        sumRef2 += deltafVec[j] * deltafVec[j];
    //    }
    //    double deltaf = sqrt(sumRef2);

    //    double warP[3][3] = { {1 + p[1], p[2], p[0]},  {p[4], 1 + p[5], p[3]}, {0, 0, 1} };
    //    double thre = 1;
    //    int Iter = 0;
    //    double Czncc = 0;
    //    double* gIntep = new double[3 * subset * subset];
    //    double* PcoordInt = new double[3 * subset * subset];
    //    while (thre > 1e-3 && Iter < maxIter || Iter == 0)
    //    {
    //        double minPcoordInt1 = subset;
    //        double minPcoordInt2 = subset;
    //        double maxPcoordInt1 = -subset;
    //        double maxPcoordInt2 = -subset;
    //        for (int j = 0; j < subset * subset; j++)
    //        {
    //            gIntep[j] = warP[0][0] * localSubHom[j] + warP[0][1] * localSubHom[j + subset * subset] + warP[0][2] * localSubHom[j + 2 * subset * subset];
    //            gIntep[j + subset * subset] = warP[1][0] * localSubHom[j] + warP[1][1] * localSubHom[j + subset * subset] + warP[1][2] * localSubHom[j + 2 * subset * subset];
    //            gIntep[j + 2 * subset * subset] = warP[2][0] * localSubHom[j] + warP[2][1] * localSubHom[j + subset * subset] + warP[2][2] * localSubHom[j + 2 * subset * subset];
    //            //cout << gIntep[j] << "              " << gIntep[j + subset * subset] << "             " << gIntep[j + 2 * subset * subset] << endl;
    //            PcoordInt[j] = pCoord[0] + gIntep[j];
    //            PcoordInt[j + subset * subset] = pCoord[1] + gIntep[j + subset * subset];
    //            PcoordInt[j + 2 * subset * subset] = pCoord[2] + gIntep[j + 2 * subset * subset] - 1;
    //            if (PcoordInt[j] < minPcoordInt1)
    //            {
    //                minPcoordInt1 = PcoordInt[j];
    //            }
    //            if (PcoordInt[j + subset * subset] < minPcoordInt2)
    //            {
    //                minPcoordInt2 = PcoordInt[j + subset * subset];
    //            }
    //            if (PcoordInt[j] > maxPcoordInt1)
    //            {
    //                maxPcoordInt1 = PcoordInt[j];
    //            }
    //            if (PcoordInt[j + subset * subset] > maxPcoordInt2)
    //            {
    //                maxPcoordInt2 = PcoordInt[j + subset * subset];
    //            }
    //        }

    //        if (minPcoordInt1 > 2 && minPcoordInt2 > 2 && maxPcoordInt1 < sizeX - 2 && maxPcoordInt2 < sizeY - 2)
    //        {
    //            int* xInt = new int[3 * subset * subset];
    //            double* deltaX = new double[3 * subset * subset];
    //            double* deltaMatX = new double[4 * subset * subset];
    //            double* deltaMatY = new double[4 * subset * subset];
    //            int* Indx = new int[16];
    //            int* D_all = new int[16];
    //            double* defIntp = new double[16];
    //            double* defIntp2 = new double[subset * subset];
    //            double sumdefIntp = 0;

    //            for (int j = 0; j < subset * subset; j++)
    //            {
    //                xInt[j] = floor(PcoordInt[j]);
    //                xInt[j + subset * subset] = floor(PcoordInt[j + subset * subset]);
    //                xInt[j + 2 * subset * subset] = floor(PcoordInt[j + 2 * subset * subset]);

    //                //cout << "j:          " << j << "          " << xInt[j] << "          " << xInt[j + subset * subset] << "          " << xInt[j + 2 * subset * subset] << endl;
    //                deltaX[j] = PcoordInt[j] - xInt[j];
    //                deltaX[j + subset * subset] = PcoordInt[j + subset * subset] - xInt[j + subset * subset];
    //                deltaX[j + 2 * subset * subset] = PcoordInt[j + 2 * subset * subset] - xInt[j + 2 * subset * subset];
    //                deltaMatX[j] = MBT[0][0] * deltaX[j] * deltaX[j] * deltaX[j] + MBT[0][1] * deltaX[j] * deltaX[j] + MBT[0][2] * deltaX[j] + MBT[0][3];
    //                deltaMatX[j + subset * subset] = MBT[1][0] * deltaX[j] * deltaX[j] * deltaX[j] + MBT[1][1] * deltaX[j] * deltaX[j] + MBT[1][2] * deltaX[j] + MBT[1][3];
    //                deltaMatX[j + 2 * subset * subset] = MBT[2][0] * deltaX[j] * deltaX[j] * deltaX[j] + MBT[2][1] * deltaX[j] * deltaX[j] + MBT[2][2] * deltaX[j] + MBT[2][3];
    //                deltaMatX[j + 3 * subset * subset] = MBT[3][0] * deltaX[j] * deltaX[j] * deltaX[j] + MBT[3][1] * deltaX[j] * deltaX[j] + MBT[3][2] * deltaX[j] + MBT[3][3];

    //                //cout << "j:          " << j << "          " << deltaMatX[j] << "              " << deltaMatX[j + subset * subset] << "             " << deltaMatX[j + 2 * subset * subset] << "              " << deltaMatX[j + 3 * subset * subset] << "             " << endl;

    //                deltaMatY[j] = MBT[0][0] * deltaX[j + subset * subset] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[0][1] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[0][2] * deltaX[j + subset * subset] + MBT[0][3];
    //                deltaMatY[j + subset * subset] = MBT[1][0] * deltaX[j + subset * subset] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[1][1] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[1][2] * deltaX[j + subset * subset] + MBT[1][3];
    //                deltaMatY[j + 2 * subset * subset] = MBT[2][0] * deltaX[j + subset * subset] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[2][1] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[2][2] * deltaX[j + subset * subset] + MBT[2][3];
    //                deltaMatY[j + 3 * subset * subset] = MBT[3][0] * deltaX[j + subset * subset] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[3][1] * deltaX[j + subset * subset] * deltaX[j + subset * subset] + MBT[3][2] * deltaX[j + subset * subset] + MBT[3][3];
    //                //cout << "j:          " << j << "          " << deltaMatY[j] << "              " << deltaMatY[j + subset * subset] << "             " << deltaMatY[j + 2 * subset * subset] << "              " << deltaMatY[j + 3 * subset * subset] << "             " << endl;
    //                Indx[0] = (xInt[j + subset * subset] - 1) * length + xInt[j];
    //                Indx[1] = (xInt[j + subset * subset]) * length + xInt[j];
    //                Indx[2] = (xInt[j + subset * subset] + 1) * length + xInt[j];
    //                Indx[3] = (xInt[j + subset * subset] + 2) * length + xInt[j];
    //                Indx[4] = (xInt[j + subset * subset] - 1) * length + xInt[j] + 1;
    //                Indx[5] = (xInt[j + subset * subset]) * length + xInt[j] + 1;
    //                Indx[6] = (xInt[j + subset * subset] + 1) * length + xInt[j] + 1;
    //                Indx[7] = (xInt[j + subset * subset] + 2) * length + xInt[j] + 1;
    //                Indx[8] = (xInt[j + subset * subset] - 1) * length + xInt[j] + 2;
    //                Indx[9] = (xInt[j + subset * subset]) * length + xInt[j] + 2;
    //                Indx[10] = (xInt[j + subset * subset] + 1) * length + xInt[j] + 2;
    //                Indx[11] = (xInt[j + subset * subset] + 2) * length + xInt[j] + 2;
    //                Indx[12] = (xInt[j + subset * subset] - 1) * length + xInt[j] + 3;
    //                Indx[13] = (xInt[j + subset * subset]) * length + xInt[j] + 3;
    //                Indx[14] = (xInt[j + subset * subset] + 1) * length + xInt[j] + 3;
    //                Indx[15] = (xInt[j + subset * subset] + 2) * length + xInt[j] + 3;
    //                //for (int ic = 0; ic < 16; ic++)
    //                //{
    //                //	cout << "j:          " << j << "          " << "ic:          " << ic << "          " << Indx[ic] << endl;
    //                //}
    //                //cout << " ***********************************************************" << endl;
    //                //cout << "j:          " << j << "          " << deltaMatY[j] << "              " << deltaMatY[j + subset * subset] << "             " << deltaMatY[j + 2 * subset * subset] << "              " << deltaMatY[j + 3 * subset * subset] << "             " << endl;
    //                D_all[0] = ImDef[(Indx[0] % length - 1) * length + Indx[0]/ length];
    //                D_all[1] = ImDef[(Indx[1] % length - 1) * length + Indx[1] / length];
    //                D_all[2] = ImDef[(Indx[2] % length - 1) * length + Indx[2] / length];
    //                D_all[3] = ImDef[(Indx[3] % length - 1) * length + Indx[3] / length];
    //                D_all[4] = ImDef[(Indx[4] % length - 1) * length + Indx[4] / length];
    //                D_all[5] = ImDef[(Indx[5] % length - 1) * length + Indx[5] / length];
    //                D_all[6] = ImDef[(Indx[6] % length - 1) * length + Indx[6] / length];
    //                D_all[7] = ImDef[(Indx[7] % length - 1) * length + Indx[7] / length];
    //                D_all[8] = ImDef[(Indx[8] % length - 1) * length + Indx[8] / length];
    //                D_all[9] = ImDef[(Indx[9] % length - 1) * length + Indx[9] / length];
    //                D_all[10] = ImDef[(Indx[10] % length - 1) * length + Indx[10] / length];
    //                D_all[11] = ImDef[(Indx[11] % length - 1) * length + Indx[11] / length];
    //                D_all[12] = ImDef[(Indx[12] % length - 1) * length + Indx[12] / length];
    //                D_all[13] = ImDef[(Indx[13] % length - 1) * length + Indx[13] / length];
    //                D_all[14] = ImDef[(Indx[14] % length - 1) * length + Indx[14] / length];
    //                D_all[15] = ImDef[(Indx[15] % length - 1) * length + Indx[15] / length];

    //                defIntp[0] = deltaMatY[j] * D_all[0] * deltaMatX[j];
    //                defIntp[1] = deltaMatY[j + subset * subset] * D_all[1] * deltaMatX[j];
    //                defIntp[2] = deltaMatY[j + 2 * subset * subset] * D_all[2] * deltaMatX[j];
    //                defIntp[3] = deltaMatY[j + 3 * subset * subset] * D_all[3] * deltaMatX[j];
    //                defIntp[4] = deltaMatY[j] * D_all[4] * deltaMatX[j + subset * subset];
    //                defIntp[5] = deltaMatY[j + subset * subset] * D_all[5] * deltaMatX[j + subset * subset];
    //                defIntp[6] = deltaMatY[j + 2 * subset * subset] * D_all[6] * deltaMatX[j + subset * subset];
    //                defIntp[7] = deltaMatY[j + 3 * subset * subset] * D_all[7] * deltaMatX[j + subset * subset];
    //                defIntp[8] = deltaMatY[j] * D_all[8] * deltaMatX[j + 2 * subset * subset];
    //                defIntp[9] = deltaMatY[j + subset * subset] * D_all[9] * deltaMatX[j + 2 * subset * subset];
    //                defIntp[10] = deltaMatY[j + 2 * subset * subset] * D_all[10] * deltaMatX[j + 2 * subset * subset];
    //                defIntp[11] = deltaMatY[j + 3 * subset * subset] * D_all[11] * deltaMatX[j + 2 * subset * subset];
    //                defIntp[12] = deltaMatY[j] * D_all[12] * deltaMatX[j + 3 * subset * subset];
    //                defIntp[13] = deltaMatY[j + subset * subset] * D_all[13] * deltaMatX[j + 3 * subset * subset];
    //                defIntp[14] = deltaMatY[j + 2 * subset * subset] * D_all[14] * deltaMatX[j + 3 * subset * subset];
    //                defIntp[15] = deltaMatY[j + 3 * subset * subset] * D_all[15] * deltaMatX[j + 3 * subset * subset];
    //                //for (int ic = 0; ic < 16; ic++)
    //                //{
    //                //	cout << "j:          " << j << "          " << "ic:          " << ic << "          " << defIntp[ic] << endl;
    //                //}
    //                //cout << " ***********************************************************" << endl;
    //                defIntp2[j] = defIntp[0] + defIntp[1] + defIntp[2] + defIntp[3] + defIntp[4] + defIntp[5] + defIntp[6] + defIntp[7] + defIntp[8] + defIntp[9] + defIntp[10] + defIntp[11] + defIntp[12] + defIntp[13] + defIntp[14] + defIntp[15];
    //                sumdefIntp += defIntp2[j];
    //            }

    //            float meandefIntp = sumdefIntp / float(subset * subset);

    //            double* deltagVec = new double[subset * subset];
    //            double sumdeltagVec = 0;
    //            for (int j = 0; j < subset * subset; j++)
    //            {
    //                deltagVec[j] = defIntp2[j] - meandefIntp;
    //                sumdeltagVec += (deltagVec[j] * deltagVec[j]);
    //            }
    //            double deltag = sqrt(sumdeltagVec);
    //            double* delta = new double[subset * subset];
    //            double deltap[6] = { 0 };
    //            for (int j = 0; j < subset * subset; j++)
    //            {
    //                delta[j] = deltafVec[j] - deltaf/ deltag * deltagVec[j];
    //                deltap[0] += -invHJacob[j] * delta[j];
    //                deltap[1] += -invHJacob[j + subset * subset] * delta[j];
    //                deltap[2] += -invHJacob[j + 2 * subset * subset] * delta[j];
    //                deltap[3] += -invHJacob[j + 3 * subset * subset] * delta[j];
    //                deltap[4] += -invHJacob[j + 4 * subset * subset] * delta[j];
    //                deltap[5] += -invHJacob[j + 5 * subset * subset] * delta[j];
    //            }
    //            deltap[0] = M[0] * deltap[0];
    //            deltap[1] = M[1] * deltap[1];
    //            deltap[2] = M[2] * deltap[2];
    //            deltap[3] = M[3] * deltap[3];
    //            deltap[4] = M[4] * deltap[4];
    //            deltap[5] = M[5] * deltap[5];

    //            double warpdelta[3][3] = { {1 + deltap[1], deltap[2], deltap[0]},  {deltap[4], 1 + deltap[5], deltap[3]}, {0, 0, 1} };
    //            double invwarpdelta[3][3] = { 0 };
    //            GetMatrixInverse(warpdelta, 3, invwarpdelta);

    //            double warP2[3][3] = { 0 };
    //            warP2[0][0] = warP[0][0] * invwarpdelta[0][0] + warP[0][1] * invwarpdelta[1][0] + warP[0][2] * invwarpdelta[2][0];
    //            warP2[0][1] = warP[0][0] * invwarpdelta[0][1] + warP[0][1] * invwarpdelta[1][1] + warP[0][2] * invwarpdelta[2][1];
    //            warP2[0][2] = warP[0][0] * invwarpdelta[0][2] + warP[0][1] * invwarpdelta[1][2] + warP[0][2] * invwarpdelta[2][2];
    //            warP2[1][0] = warP[1][0] * invwarpdelta[0][0] + warP[1][1] * invwarpdelta[1][0] + warP[1][2] * invwarpdelta[2][0];
    //            warP2[1][1] = warP[1][0] * invwarpdelta[0][1] + warP[1][1] * invwarpdelta[1][1] + warP[1][2] * invwarpdelta[2][1];
    //            warP2[1][2] = warP[1][0] * invwarpdelta[0][2] + warP[1][1] * invwarpdelta[1][2] + warP[1][2] * invwarpdelta[2][2];
    //            warP2[2][0] = warP[2][0] * invwarpdelta[0][0] + warP[2][1] * invwarpdelta[1][0] + warP[2][2] * invwarpdelta[2][0];
    //            warP2[2][1] = warP[2][0] * invwarpdelta[0][1] + warP[2][1] * invwarpdelta[1][1] + warP[2][2] * invwarpdelta[2][1];
    //            warP2[2][2] = warP[2][0] * invwarpdelta[0][2] + warP[2][1] * invwarpdelta[1][2] + warP[2][2] * invwarpdelta[2][2];

    //            warP[0][0] = warP2[0][0], warP[0][1] = warP2[0][1], warP[0][2] = warP2[0][2];
    //            warP[1][0] = warP2[1][0], warP[1][1] = warP2[1][1], warP[1][2] = warP2[1][2];
    //            warP[2][0] = warP2[2][0], warP[2][1] = warP2[2][1], warP[2][2] = warP2[2][2];

    //            //cout << warP[0][0] << "        " << warP[0][1] << "        " << warP[0][2] << endl;
    //            //cout << warP[1][0] << "        " << warP[1][1] << "        " << warP[1][2] << endl;
    //            //cout << warP[2][0] << "        " << warP[2][1] << "        " << warP[2][2] << endl;

    //            thre = sqrt(deltap[0] * deltap[0] + deltap[3] * deltap[3]);
    //            //cout << "thre:              " << thre << endl;
    //            Iter++;
    //            p[0] = warP[0][2];
    //            p[1] = warP[0][0] - 1;
    //            p[2] = warP[0][1];
    //            p[3] = warP[1][2];
    //            p[4] = warP[1][0];
    //            p[5] = warP[1][1] - 1;
    //            //cout << p[0] << "        " << p[1] << "        " << p[2] << "        " << p[3] << "        " << p[4] << "        " << p[5] <<  endl;
    //            dispBias[0] = p[0];
    //            dispBias[1] = p[3];
    //            double Cznssd = 0;
    //            for (int j = 0; j < subset * subset; j++)
    //            {
    //                double deltafg = (deltafVec[j] / deltaf - deltagVec[j] / deltag);
    //                Cznssd += deltafg * deltafg;
    //            }
    //            Czncc = 1 - 0.5 * Cznssd;
    //            delete[] deltagVec;
    //            delete[] defIntp2;
    //            delete[] defIntp;
    //            delete[] D_all;
    //            delete[] Indx;
    //            delete[] deltaMatY;
    //            delete[] deltaMatX;
    //            delete[] deltaX;
    //            delete[] delta;
    //            delete[] xInt;
    //        }
    //        else
    //        {
    //            Iter = Iter + 1;
    //            p[0] = 0, p[1] = 0, p[2] = 0, p[3] = 0, p[4] = 0, p[5] = 0;
    //            dispBias[0] = 0;
    //            dispBias[1] = 0;
    //            Czncc = -1;
    //            break;
    //        }
    //    }
    //    delete[] gIntep;
    //    delete[] PcoordInt;
    //    delete[] invHJacob;
    //    delete[] deltafVec;
    //    delete[] fSubset;
    //    delete[] Jacobian;
    //    //delete[] nablafxy;
    //    //delete[] nablafyy;
    //    //delete[] nablafxx;
    //    delete[] nablafy;
    //    delete[] nablafx;
    return 1;
}
