#include "color.h"

#define SHOW 1
/*****
 *      黑      灰      白      红       橙      黄     绿      青      蓝      紫
LowH    0       0       0      0/156    11     26     35      78     100     125
HighH   180     180     180    10/180   25     34     77      99     124     155
LowS    0       0       0      43       43     43
HighS   255     43      30     255      255    255
LowV    0       46      221    46       46     46
HighV   46      220     255    255      255    255

*****/
int iLowH = 100;
int iHighH = 140;
    
int iLowS = 90; 
int iHighS = 255;
    
int iLowV = 90;
int iHighV = 255;

int main( int argc, char** argv )
{
    VideoCapture cap(0); //capture the video from web cam
 
    if ( !cap.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the web cam" << endl;
         return -1;
    }
 
    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
    

    
    //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighH, 179);
    
    cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);
    
    cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);
 
    while (true)
    {
        Mat imgOriginal;
 
        bool bSuccess = cap.read(imgOriginal); // read a new frame from video
 
        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
#if 0
        Mat imgHSV;
        vector<Mat> hsvSplit;
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
        
        //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
        split(imgHSV, hsvSplit);
        equalizeHist(hsvSplit[2],hsvSplit[2]);
        merge(hsvSplit,imgHSV);
        Mat imgThresholded;
        
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
        
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
                imshow("output", output);
            }   
        }
        imshow("Thresholded Image", imgThresholded); //show the thresholded image
        imshow("Original", imgOriginal); //show the original image
#else        
        colorDetection(imgOriginal);
#endif
    
        char key = (char) waitKey(300);
        if(key == 27)
            break;

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

    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
    
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
#if SHOW
            imshow("output", output);
#endif
        }   
    }
#if SHOW
    imshow("Thresholded Image", imgThresholded); //show the thresholded image
    imshow("Original", imgOriginal); //show the original image
#endif
    return output;
}