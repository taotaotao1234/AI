#ifndef _COLOR_H
#define _COLOR_H
#include <stdio.h>
#include <iostream>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp> 
#include <opencv4/opencv2/highgui/highgui_c.h>
#include <string.h>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat colorDetection(Mat imgOriginal); 
Mat characterDetection(Mat frame);
Scalar getMSSIM(Mat  inputimage1, Mat inputimage2);
unsigned int ahash(Mat src1,Mat src2 );
Mat qrcode(Mat img);

#endif