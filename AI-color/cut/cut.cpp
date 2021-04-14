#include <stdio.h>
#include <iostream>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp> 
#include <opencv4/opencv2/highgui/highgui_c.h>

using namespace cv;
using namespace std;
Mat colorDetection(Mat imgOriginal);
Mat characterDetection(Mat frame);

Mat output1;
Mat output2;

int main( int argc, char** argv )
{
    cv::Mat src = cv::imread(argv[1]);

    Mat dst = colorDetection(src);

    char picture_path[128] = {0};
    char picture_path_temp[128] = {0};
    sprintf(picture_path,"%s",argv[1]);
    //printf("%s\n",picture_path);
    unsigned char path_flag = 0;
    unsigned char path_count = 0;
    for(int i=128; i>=0; i--)
    {
        if(picture_path[i]== 0)
        {
            continue;
        }
        else if(picture_path[i] == '.' && path_flag == 0)
        {
            path_flag = 1;
            continue;
        }
        if(path_flag ==1)
        {
            picture_path_temp[i] = picture_path[i];
            path_count++;
        }
    }
    printf("%s\n",picture_path_temp);


    string output2_name = format("%s-2.jpg",picture_path_temp); 
    string output1_name = format("%s-1.jpg",picture_path_temp); 

    if(!output2.empty())
    {
        resize(output2, output2, Size(100, 100));
        cv::imwrite(output2_name,output2); 
    }
    
    if(!output1.empty())
    {
        Mat output = characterDetection(output1);

        cv::imwrite(output1_name,output); 
    }

    return 0;
}


Mat colorDetection(Mat imgOriginal)
{
 
    Mat imgHSV;
    vector<Mat> hsvSplit;
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    
    //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
    split(imgHSV, hsvSplit);
    equalizeHist(hsvSplit[2],hsvSplit[2]);
    merge(hsvSplit,imgHSV);
    Mat imgThresholded;
    // Red 156 180 43 255 46 255 
    //inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
    inRange(imgHSV, Scalar(91, 220, 0), Scalar(180, 225, 255), imgThresholded); 

    //开操作 (去除一些噪点)
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
    
    //闭操作 (连接一些连通域)
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

    //边沿检测，检测最外层轮廓 
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    findContours(imgThresholded, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<Rect> boundRect(contours.size()); //定义外接矩形集
    int x0=0, y0=0, w0=0, h0=0;
    Mat output;
    for(int i=0; i<contours.size(); i++)
    {
        boundRect[i] = boundingRect((Mat)contours[i]); //查找每个轮廓的外接矩形
        
        x0 = boundRect[i].x;  //获得第i个外接矩形的左上角的x坐标
        y0 = boundRect[i].y; //获得第i个外接矩形的左上角的y坐标
        w0 = boundRect[i].width; //获得第i个外接矩形的宽度
        h0 = boundRect[i].height; //获得第i个外接矩形的高度
        //筛选  去除边框检测矩形
        if(w0>30 && h0>30)
        {
            rectangle(imgOriginal, Point(x0, y0), Point(x0+w0, y0+h0), Scalar(0, 255, 0), 2, 8); //绘制第i个外接矩形
            output = imgOriginal(boundRect[i]);
            output1 = imgOriginal(boundRect[i]);
            output2 = imgThresholded(boundRect[i]);

            imshow("output1",output1);
            imshow("output2",output2);
        }   
    }
    imshow("Thresholded Image", imgThresholded); //show the thresholded image
    imshow("Original", imgOriginal); //show the original image
    char key = (char) waitKey(0);

    return output;
}

Mat characterDetection(Mat frame)
{
    Mat input = frame.clone();

    resize(input, input, Size(100, 100));

    Mat imgHSV;
    vector<Mat> hsvSplit;
    cvtColor(input, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    
    //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
    split(imgHSV, hsvSplit);
    equalizeHist(hsvSplit[2],hsvSplit[2]);
    merge(hsvSplit,imgHSV);
    Mat imgThresholded;

        
    //inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
    // white 0 180 0 30 221 255 
    inRange(imgHSV, Scalar(0, 0, 221), Scalar(180, 30, 255), imgThresholded); //Threshold the image


    //开操作 (去除一些噪点)
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
    
    //闭操作 (连接一些连通域)
    element = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
    
    return imgThresholded;
}  