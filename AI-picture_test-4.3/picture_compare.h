#ifndef __PICTURE_COMPARE_H__
#define __PICTURE_COMPARE_H__
#include <iostream>
#include "opencv2/opencv.hpp" 
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include <math.h>
#include "http_updata.h"

#include <stdio.h>

using namespace std;
using namespace cv;

unsigned int ahash(Mat src1, Mat src2 );
cv::Point template_match(Mat img1, Mat img2);
Scalar getMSSIM(Mat inputimage1, Mat inputimage2);
void unevenLightCompensate(Mat &image, int blockSize);
unsigned char detection(Mat src,Mat dst);

typedef struct{
    unsigned int point_x;
    unsigned int point_y;
    unsigned int height;
    unsigned int width;
}Dest_Parameters;

typedef struct{
    float hash_Result;
    float ssim_Result;
    float template_Result;
    short different_Areas;
    Mat   picture_Result;
    short picture_choice;
    unsigned char type_Result;
}Final_Result;

typedef struct{
    Point sqdiff_point;
    Point sqdiff_normed_point;
    Point ccorr_point;
    Point ccorr_normed_point;
    Point ccoeff_point;
    Point ccoeff_normed_point;

    Point result_point;
    unsigned char same_point;
}Template_Result;

typedef struct{
    float ssim_Result;
    short argc;
}SSIM_Result;

#endif