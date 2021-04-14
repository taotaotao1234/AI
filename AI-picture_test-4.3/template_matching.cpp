#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

int main(int argc,char **argv)
{
//步骤一：读取图片
    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);

    cvtColor(img1, img1, COLOR_BGR2GRAY);
    cvtColor(img2, img2, COLOR_BGR2GRAY);
    //cv::imshow("【被查找的图像】", img1);
    //cv::imshow("【模版图像】", img2);

//步骤二：创建一个空画布用来绘制匹配结果
    cv::Mat dstImg0;cv::Mat dstImg1;cv::Mat dstImg2;cv::Mat dstImg3;cv::Mat dstImg4;
    dstImg1.create(img1.dims,img1.size,img1.type());
    //cv::imshow("createImg",dstImg);

//步骤三：匹配，最后一个参数为匹配方式，共有6种，详细请查阅函数介绍
    cv::matchTemplate(img1, img2, dstImg0, 0);
    cv::matchTemplate(img1, img2, dstImg1, 1);
    cv::matchTemplate(img1, img2, dstImg2, 3);
    cv::matchTemplate(img1, img2, dstImg3, 4);
    cv::matchTemplate(img1, img2, dstImg4, 5);
    
//步骤四：归一化图像矩阵，可省略
    cv::normalize(dstImg0, dstImg0, 0, 1, 32);
    cv::normalize(dstImg1, dstImg1, 0, 1, 32);
    cv::normalize(dstImg2, dstImg2, 0, 1, 32);
    cv::normalize(dstImg3, dstImg3, 0, 1, 32);
    cv::normalize(dstImg4, dstImg4, 0, 1, 32);

//步骤五：获取最大或最小匹配系数
//首先是从得到的 输出矩阵中得到 最大或最小值（平方差匹配方式是越小越好，所以在这种方式下，找到最小位置）
//找矩阵的最小位置的函数是 minMaxLoc函数
    cv::Point minPoint[5];
    cv::Point maxPoint[5];
    double *minVal = 0;
    double *maxVal = 0;
    cv::minMaxLoc(dstImg0, minVal, maxVal, &minPoint[0],&maxPoint[0]);
    cout <<"0: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; minPoint="<<minPoint[0]<<"; maxPoint="<<maxPoint[0]<<endl;
    cv::minMaxLoc(dstImg1, minVal, maxVal, &minPoint[1],&maxPoint[1]);
    cout <<"1: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; minPoint="<<minPoint[1]<<"; maxPoint="<<maxPoint[1]<<endl;
    cv::minMaxLoc(dstImg2, minVal, maxVal, &minPoint[2],&maxPoint[2]);
    cout <<"2: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; maxPoint="<<maxPoint[2]<<"; minPoint="<<minPoint[2]<<endl;
    cv::minMaxLoc(dstImg3, minVal, maxVal, &minPoint[3],&maxPoint[3]);
    cout <<"3: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; maxPoint="<<maxPoint[3]<<"; minPoint="<<minPoint[3]<<endl;
    cv::minMaxLoc(dstImg4, minVal, maxVal, &minPoint[4],&maxPoint[4]);
    cout <<"4: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; maxPoint="<<maxPoint[4]<<"; minPoint="<<minPoint[4]<<endl;

//步骤六：开始正式绘制
    unsigned int count = 0; 
    unsigned char count0 = 0; unsigned char count1 = 0; unsigned char count2 = 0; unsigned char count3 = 0; 
    unsigned int failed_count = 0;
    for(int i=0;i<5;i++){
        for(int j=4;j>i;j--){
            if((minPoint[i].x==minPoint[j].x && minPoint[i].y==minPoint[j].y) || (minPoint[i].x==maxPoint[j].x && minPoint[i].y==maxPoint[j].y) || (maxPoint[i].x==maxPoint[j].x && maxPoint[i].y==maxPoint[j].y))
            {
                if(i==0)
                    count0 ++;
                else if(i==1)
                    count1++;
                else if(i==2)
                    count2++;
                else if(i==3)
                    count3++;
                else
                {}
            }
            else{
                failed_count++;
                //cout << "i="<<i<<"; j="<<j<<endl;
            }
        }
    }
    count = count0 + count1 + count2 + count3;
    cout << "count=" << count << "; failed count="<< failed_count << endl;
    cv::Point bestPoint;
    if(count >= 3)
    {
        if(count0 >= 2){
            bestPoint.x = minPoint[0].x;
            bestPoint.y = minPoint[0].y;
        }
        else if(count1 >= 2){
            bestPoint.x = minPoint[1].x;
            bestPoint.y = minPoint[1].y;
        }
        else if(count2 >= 2){
            bestPoint.x = maxPoint[2].x;
            bestPoint.y = maxPoint[2].y;
        }
        else if(count3 >= 2){
            bestPoint.x = maxPoint[3].x;
            bestPoint.y = maxPoint[3].y;
        }
        cout << "best point = [" << bestPoint.x << "," << bestPoint.y << "]" << endl;
        cv::rectangle(img1, bestPoint, cv::Point(bestPoint.x + img2.cols, bestPoint.y + img2.rows), cv::Scalar(0,255,0), 2, 8);
        cv::imshow("【匹配后的图像0】", img1);
        cv::rectangle(dstImg0, bestPoint, cv::Point(bestPoint.x + img2.cols, bestPoint.y + img2.rows), cv::Scalar(0,0,0), 3, 8);
        cv::imshow("【匹配后的计算过程图像】", dstImg1);
        cv::waitKey(0);
        return 0;
    }
    else
    {
        cout << "Not a similar picture "<< endl;
        return -1;
    }
}