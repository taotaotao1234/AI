#include "picture_compare.h"
#define output 0
#define Hash_min_standards 60
#define Cut_picture_x Picture_max_x/10
#define Cut_picture_y Picture_max_y/10
#define Location_differences 5 // 不同模板匹配算法之间，结论点之间最大差值
short Partition_num = 8;
short pic_diff_num = 0;
int Picture_max_x = 960*2;
int Picture_max_y = 540*2;
float Zoom_factor_x = 0;
float Zoom_factor_y = 0;
char picture_path_temp[128] = {0};
Final_Result Final;
Final_Result Det;
Template_Result Tempalte;
cv::Mat result_img;
Point src_point;
char param3[128] = {0};
unsigned char shart_argc = 2; 
short basic_Picture_Count = 0;
char choice_src_addr[128] = {0};    // 选择基准图的路径

int main(int argc, char ** argv)
{
    struct timeval tv_begin, tv_end;
    gettimeofday(&tv_begin, NULL);
    if(argc < 2)
    {
        cout << "Please input compare picture address" << endl;
        upload_cur_http(Partition_num*Partition_num+4);
        return -1;
    }
    cv::Mat dst_img = cv::imread(argv[1]);
    result_img = cv::imread(argv[1]);   // 测试图
    char picture_path[128] = {0};   
    //sprintf(param3,"%s",argv[1]);
    sprintf(picture_path,"%s",argv[1]);
    //printf("%s\n",picture_path);
 /*
  *     裁剪信息中的对比图片路径  picture_path_temp
  */   
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
#if output
    printf("%s\n",picture_path_temp);
#endif
    int picture_num = shart_argc;
    short choice_num = 0;

    Final.ssim_Result = 1;

/*
 *   1. 基准图SSIM算法相似度计算
 *      记录最相似图片信息
 */ 
    float all_SSIM[256] = {0};
    short all_SSIM_count = 0;
    while(1)
    {
        if(argv[picture_num] == NULL )
        {
            sprintf(param3,"%s",argv[picture_num-1]);
#if output
            printf("no more picture; id = %s\n",argv[picture_num-1]);
#endif
            break;
        }
        cv::Mat src_img_dect = cv::imread(argv[picture_num]);
        // if(src_img_dect.empty())
        // {
        //     continue;
        // }
        if(src_img_dect.cols != dst_img.cols)
        {
#if output
            printf("Different picture sizes \n");
#endif
            picture_num ++;
            all_SSIM[all_SSIM_count] = 0;
            all_SSIM_count ++;
            continue;
        }
        Scalar SSIM_dect = getMSSIM(dst_img, src_img_dect);
        float result_ssim_dect = 0;
        result_ssim_dect = (SSIM_dect.val[2] + SSIM_dect.val[1] + SSIM_dect.val[0])/3 * 100;
    // 记录每次比较结果
        all_SSIM[all_SSIM_count] = result_ssim_dect;
        all_SSIM_count ++;
    // 记录最相似结果
        if(Final.ssim_Result  < result_ssim_dect)
        {
            Final.ssim_Result = result_ssim_dect;
            Final.picture_choice = picture_num;
            choice_num = Final.picture_choice ;
        }
#if output
        printf("result = [%.2f]\n",result_ssim_dect);
#endif
        picture_num ++;
    }
/*
 *   2. 选择最相似的基准图先做比较
 *      若相似度为100,则不做后续比较; 若存在差异，则继续进行后续比较
 */ 
 
    Final.template_Result = 0;
    Final.type_Result = 1;
    Det.template_Result = 100;
    cv::Mat src_img = cv::imread(argv[Final.picture_choice]);
#if output
    printf("Use picture = [%s]\n",argv[Final.picture_choice]);
#endif
    unsigned char temp_type = 1;
    temp_type = detection(src_img,dst_img);
    Final.type_Result = temp_type;

    Final.template_Result = Det.template_Result;
    Final.picture_Result = result_img;
    Final.different_Areas = pic_diff_num;

    if(Det.template_Result == 100 && temp_type != 1)
    {
        // 相似度100,不进行其他操作
#if output
        printf("SSIM Most = [%s]\n",argv[Final.picture_choice]);
#endif
    }
    else
    {
/*
 *   3. 对SSIM算法求出的相似度进行排序，找到与最大相似度差距不大的图片信息，由大到小依次比较
 *      若相似度为100,则不做后续比较; 若存在差异，则继续进行后续比较，直到没有相似图片
 */ 
    // 如果得出的结论是图片有差异，再次选用相似
        basic_Picture_Count = all_SSIM_count;
        short choice_ssim[256] = {0};
        short choice_ssim_count = 0;
        float all_SSIM_temp[256] = {0};
        float ssim_temp = 0;
        short all_SSIM_count_temp = all_SSIM_count;
        for(int i=0;i<all_SSIM_count;i++)
        {
            all_SSIM_temp[i] = all_SSIM[i];
        }

    // 相似度排序， 数组all_SSIM[]记录每张基础图相似度,有小到大冒泡排序
        for(int i=0; i<all_SSIM_count-1; i++)
        {
            for(int j=0; j<all_SSIM_count-1-i; j++)
            {
                if(all_SSIM_temp[j] > all_SSIM_temp[j+1])
                {
                    ssim_temp = all_SSIM_temp[j];
                    all_SSIM_temp[j] = all_SSIM_temp[j+1];
                    all_SSIM_temp[j+1] = ssim_temp;
                }
            }            
        }
#if output        
        printf("all_SSIM_temp: ");
        for(int i=0;i<all_SSIM_count;i++)
        {
            if(i%10 == 0)
            {
                printf(" \n");
            }
            printf("%.2f \t",all_SSIM_temp[i]);
        }
        printf(" \n");
#endif
        all_SSIM_count --;  
        unsigned char find_Picture_flag = 0;
        while(all_SSIM_count--)
        {
            if(find_Picture_flag != 0)
            {
                break;
            }
            if((all_SSIM_temp[all_SSIM_count] > Final.ssim_Result - 2 || all_SSIM_temp[all_SSIM_count] > 40)&& all_SSIM_temp[all_SSIM_count] > 10)
            {
                //printf("most_num = %d; now_num = %d\n",choice_num,all_SSIM_count);
                for(int i=0; i< all_SSIM_count_temp; i++)
                {
                    float temp_ssim =  abs(all_SSIM_temp[all_SSIM_count] - all_SSIM[i]);
                    if(temp_ssim < 0.000001)
                    {
                        // i 为图片信息地址
        #if output
                        printf("most_num = %d; now_num = %d\n",choice_num,i + shart_argc);
                        printf("Use picture = [%s],ssim = %.2f\n",argv[i + shart_argc],all_SSIM_temp[all_SSIM_count]);
        #endif
                        // 记录最相似的图片组 和 相似度大于40的图片组, 排除已经比较的最相似的图
                        cv::Mat src_img_dect = cv::imread(argv[i + shart_argc]);
                        // 裁剪，比较
                        unsigned char temp_type = 0;
                        temp_type = detection(src_img_dect,dst_img);

                        if(Det.template_Result >= 99.9 && temp_type != 1)
                        {
                            // 相似度100,不进行其他操作
                            Final.template_Result = Det.template_Result;
                            Final.picture_choice = i + shart_argc;
                            Final.different_Areas = 0;
                            Final.picture_Result = result_img;
                            Final.type_Result = temp_type;
                            find_Picture_flag = 1;
        #if output
                            printf("SSIM result: %d; Det.template_Result = %.2f\n", i + shart_argc, Det.template_Result);
        #endif
                            break;
                        }
                        else
                        {
                            // 比较图组下一张图片
        #if output
                            printf("next picture: %d; Det.template_Result = %.2f\n", i + shart_argc, Det.template_Result);
        #endif
                            if(Det.template_Result > Final.template_Result)
                            {
                                Final.template_Result = Det.template_Result;
                                Final.picture_choice = i + shart_argc;
                                Final.different_Areas = pic_diff_num;
                                Final.picture_Result = result_img;
                                Final.type_Result = temp_type;
                            }
                        }                        
                    }
                }
            }
            else
            {
                break;
            }
        }
    }
//    4. 得出图片相似度结论

//    5.结论图片边缘虚化处理
//        1> 全部图虚化，然后使用图像混合，将对比区域在虚化图上覆盖
//        2> 对虚化部分进行像素值加减法，变深或者变暗
    // 记录选择的基准图
//    src_img = cv::imread(argv[Final.picture_choice]);
    sprintf(choice_src_addr,"%s",argv[Final.picture_choice]);
    cv::rectangle(Final.picture_Result, cv::Point((Cut_picture_x)*Zoom_factor_x,(Cut_picture_y)*Zoom_factor_y), cv::Point((Picture_max_x-Cut_picture_x)*Zoom_factor_x, (Picture_max_y-Cut_picture_y)*Zoom_factor_y), cv::Scalar(255,255,255), 4, 8);
//    cv::rectangle(src_img, cv::Point((src_point.x)*Zoom_factor_x,(src_point.y)*Zoom_factor_y), cv::Point((Picture_max_x-Cut_picture_x*2+src_point.x)*Zoom_factor_x, (Picture_max_y-Cut_picture_y*2+src_point.y)*Zoom_factor_y), cv::Scalar(255,255,255), 4, 8);
/*
    for(int m=0; m<src_img.rows; m++)
    {
        for(int v=0; v<src_img.cols; v++)
        {
            if(src_point.x < v && v<Picture_max_x-Cut_picture_x*2+src_point.x  &&  src_point.y < m && m<Picture_max_y-Cut_picture_y*2+src_point.y)
            {
                continue;
            }
            src_img.at<Vec3b>(m,v)[0] = src_img.at<Vec3b>(m,v)[0]/2;
            src_img.at<Vec3b>(m,v)[1] = src_img.at<Vec3b>(m,v)[1]/2;
            src_img.at<Vec3b>(m,v)[2] = src_img.at<Vec3b>(m,v)[2]/2;

        }
    }
*/
    for(int m=0; m<Final.picture_Result.rows; m++)
    {
        for(int v=0; v<Final.picture_Result.cols; v++)
        {
            if(Cut_picture_x < v && v<Picture_max_x-Cut_picture_x  &&  Cut_picture_y < m && m<Picture_max_y-Cut_picture_y)
            {
                continue;
            }

            Final.picture_Result.at<Vec3b>(m,v)[0] = Final.picture_Result.at<Vec3b>(m,v)[0]/2;
            Final.picture_Result.at<Vec3b>(m,v)[1] = Final.picture_Result.at<Vec3b>(m,v)[1]/2;
            Final.picture_Result.at<Vec3b>(m,v)[2] = Final.picture_Result.at<Vec3b>(m,v)[2]/2;
        }
    }


//    7.输出结论图片
    
    string filename_dst = format("%s_cp.jpg", picture_path_temp); 
    cv::imwrite(filename_dst,Final.picture_Result); 
    //string filename_src = format("%s_src.jpg", picture_path_temp); 
    //cv::imwrite(filename_src,src_img); 
    
    // 结果类型判断
    if(Final.type_Result == 0)
    {
        // 正常情况
        printf("SSIM = %0.2f, Template = %0.2f\n",Final.ssim_Result, Final.template_Result);
        upload_cur_http(Final.different_Areas);
    }
    else
    {
        // 错位严重，HASH差别过大
        printf("\nno same picture\n");
        Final.different_Areas = Partition_num*Partition_num + 1;
        upload_cur_http(Final.different_Areas);
    }
    
    sleep(1);
    gettimeofday(&tv_end, NULL);
    if(tv_end.tv_sec < tv_begin.tv_sec)
        tv_end.tv_sec += 60;
    int s = tv_end.tv_sec - tv_begin.tv_sec;

    if(tv_end.tv_usec < tv_begin.tv_usec)
    {
        s -= 1;
        tv_end.tv_usec += 1000*1000;
    }
    int us = tv_end.tv_usec - tv_begin.tv_usec;
#if output
    printf("Time  [%d] [%d]\n",s,us);
#endif
    return 0;
}

// 均值 hash 算法
unsigned int ahash(Mat src1,Mat src2 )
{ 
    if (!src1.data|| !src2.data)
    {
        cout << "ERROR : could not load image.";
        return -1;
    }
//图片缩小为16x16大小
    resize(src1, src1, Size(16, 16), 0, 0, INTER_CUBIC);
    resize(src2, src2, Size(16, 16), 0, 0, INTER_CUBIC);
//图片转换为灰度
    cvtColor(src1, src1, COLOR_BGR2GRAY);
    cvtColor(src2, src2, COLOR_BGR2GRAY);
	int iAvg1 = 0, iAvg2 = 0;
	int arr1[256], arr2[256];
	for (int i = 0; i < 16; i++) 
    {
		uchar* data1 = src1.ptr<uchar>(i);
		uchar* data2 = src2.ptr<uchar>(i);
		int tmp = i * 16;
		for (int j = 0; j < 16; j++) 
        {
			int tmp1 = tmp + j;
			arr1[tmp1] = data1[j] / 4 * 4;
			arr2[tmp1] = data2[j] / 4 * 4;
			iAvg1 += arr1[tmp1];
			iAvg2 += arr2[tmp1];
		}
	}
	iAvg1 /= 256;
	iAvg2 /= 256;
	for (int i = 0; i < 256; i++) {
		arr1[i] = (arr1[i] >= iAvg1) ? 1 : 0;
		arr2[i] = (arr2[i] >= iAvg2) ? 1 : 0;
	}
	int iDiffNum = 0;
	for (int i = 0; i < 256; i++)
		if (arr1[i] != arr2[i])
			++iDiffNum;
    return iDiffNum;
}

cv::Point template_match(Mat img1,Mat img2)
{
    cv::Point bestPoint = {0,0};
    if (!img1.data|| !img2.data)
    {
        cout << "ERROR : could not load image.";
        return bestPoint;
    }
//步骤一：读取图片
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
    cv::Point bestPoint_temp[5];

    double *minVal = 0;
    double *maxVal = 0;
    cv::minMaxLoc(dstImg0, minVal, maxVal, &minPoint[0],&maxPoint[0]);
    bestPoint_temp[0] = minPoint[0];
    Tempalte.sqdiff_point = minPoint[0];
    //cout <<"0: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; minPoint="<<minPoint[0]<<"; maxPoint="<<maxPoint[0]<<endl;
    
    cv::minMaxLoc(dstImg1, minVal, maxVal, &minPoint[1],&maxPoint[1]);
    bestPoint_temp[1] = minPoint[1];
    Tempalte.sqdiff_normed_point = minPoint[1];
    //cout <<"1: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; minPoint="<<minPoint[1]<<"; maxPoint="<<maxPoint[1]<<endl;
    
    cv::minMaxLoc(dstImg2, minVal, maxVal, &minPoint[2],&maxPoint[2]);
    bestPoint_temp[2] = maxPoint[2];
    Tempalte.ccorr_normed_point = maxPoint[2];
    //cout <<"2: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; maxPoint="<<maxPoint[2]<<"; minPoint="<<minPoint[2]<<endl;
    
    cv::minMaxLoc(dstImg3, minVal, maxVal, &minPoint[3],&maxPoint[3]);
    bestPoint_temp[3] = maxPoint[3];
    Tempalte.ccoeff_point = maxPoint[3];
    //cout <<"3: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; maxPoint="<<maxPoint[3]<<"; minPoint="<<minPoint[3]<<endl;
    
    cv::minMaxLoc(dstImg4, minVal, maxVal, &minPoint[4],&maxPoint[4]);
    bestPoint_temp[4] = maxPoint[4];
    Tempalte.ccoeff_normed_point = maxPoint[4];
    //cout <<"4: "<<"minVal="<<minVal<< "; maxVal="<<maxVal<<"; maxPoint="<<maxPoint[4]<<"; minPoint="<<minPoint[4]<<endl;
    

//步骤六：开始正式绘制
    unsigned int count = 0; 
    unsigned char count0 = 0; unsigned char count1 = 0; unsigned char count2 = 0; unsigned char count3 = 0; 
    unsigned char point_count[10]= {0};
    unsigned int failed_count = 0;
    for(int i=0; i<5; i++)
    {
        for(int j=4; j>i; j--)
        {
           if( bestPoint_temp[i].x >= bestPoint_temp[j].x - Location_differences && bestPoint_temp[i].x <= bestPoint_temp[j].x + Location_differences
            && bestPoint_temp[i].y >= bestPoint_temp[j].y - Location_differences && bestPoint_temp[i].y <= bestPoint_temp[j].y + Location_differences)
            {
                if(bestPoint_temp[i].x == 0 && bestPoint_temp[i].y == 0)
                    continue;
                if(i==0)
                    point_count[0] ++;
                else if(i==1)
                    point_count[1] ++;
                else if(i==2)
                    point_count[2] ++;
                else if(i==3)
                    point_count[3] ++; 
                else
                {}
            }
            else
            {
                failed_count++;
                //cout << "i="<<i<<"; j="<<j<<endl;
            }
        }
    }
    
    count = point_count[0] + point_count[1] + point_count[2] + point_count[3];
    //cout << "count=" << count << "; failed count="<< failed_count << endl;
    if(count >= 1)
    {
        for(int i=0; i<3; i++){
            for(int j=i; j<3; j++){
                if(point_count[i] > point_count[j]){
                    Tempalte.same_point = point_count[i];
                    count0 = i;
                }else{
                    Tempalte.same_point = point_count[j];
                    count0 = j;
                }
            }
        }
        if(count0 == 0){
            bestPoint = bestPoint_temp[0];
        }
        else if(count0 == 1){
            bestPoint = bestPoint_temp[1];
        }
        else if(count0 == 2){
            bestPoint = bestPoint_temp[2];
        }
        else if(count0 == 3){
            bestPoint = bestPoint_temp[3];
        }

    }
    else
    {
        //cout << "Not a similar picture "<<"[template match] "<< endl;
        bestPoint.x = 3000;
        bestPoint.y = 3000;
    }
    Tempalte.same_point = count + 1;
    Tempalte.result_point = bestPoint;
    return bestPoint;
}

// SSIM 算法
Scalar getMSSIM(Mat  inputimage1, Mat inputimage2)
{
    Mat i1 = inputimage1;
    Mat i2 = inputimage2;
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I2_2 = I2.mul(I2);
    Mat I1_2 = I1.mul(I1);
    Mat I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);
    Mat ssim_map;
    divide(t3, t1, ssim_map);
    Scalar mssim = mean(ssim_map);
    return mssim;
}


unsigned char detection(Mat src_img,Mat dst_img)
{
    Dest_Parameters dst;
    Dest_Parameters src;
    Point cut_src = {0,0};
    char picture_path[128] = {0};
    char picture_path_temp[128] = {0};
    pic_diff_num = 0;
    result_img = dst_img.clone();
// 根据图像大小
    if(src_img.cols > 384)
    {
        Zoom_factor_x = (float)(src_img.cols *10000/ Picture_max_x) / 10000;
        Zoom_factor_y = (float)(src_img.rows *10000/ Picture_max_y) / 10000;
        //printf("Zoom_factor_x=%f, Zoom_factor_y=%f\n", Zoom_factor_x, Zoom_factor_y);
        resize(src_img, src_img, Size(Picture_max_x, Picture_max_y), 0, 0, INTER_CUBIC);
        resize(dst_img, dst_img, Size(Picture_max_x, Picture_max_y), 0, 0, INTER_CUBIC);
    }
    else
    {
        Zoom_factor_x = 1;
        Zoom_factor_y = 1;
        Picture_max_x = src_img.cols;
        Picture_max_y = src_img.rows;
    }
    
    float result = 0;
    cv::Point result_point;
/*
    1. 利用hash算法，判断是否为同一图片，或同一图片改变了色调，同一图片直接返回图片相似度结论
*/
    result = 100 - (float)ahash(src_img,dst_img)*100/256;
    Det.hash_Result = result;
    if(result >= 99.9)
    {
        cout << "Same picture "<< " [hash]" << endl;
        return 0;
    }
    else if(result < Hash_min_standards)
    {
        cout << "picture low similarity"<< " [hash]" << endl;
        return 1;
    }

    cv::Mat dst_img_temp = dst_img.clone();
    cv::Mat src_img_temp = src_img.clone();               
// SSIM
    blur(dst_img_temp,dst_img_temp,Size(10,10));
    blur(src_img_temp,src_img_temp,Size(10,10));
    Scalar SSIM1 = getMSSIM(dst_img_temp, src_img_temp);
    float result_ssim = (SSIM1.val[2] + SSIM1.val[1] + SSIM1.val[0])/3 * 100;
    Det.ssim_Result = result_ssim;
    if(result_ssim > 90)
    {
        // 待测图片与原图相同位置图片，相似度高，得出结论，存在此图 
        
    }  
    //cout << "Picture compare result =" << result_ssim << " [SSIM]" << endl;

/*
    2.1 图片去边（去除错位位置），错位情况判断
*/
    // 左上角全局判断错位
    cv::Rect cut_dst_img_rect(Cut_picture_x,Cut_picture_y,Picture_max_x-Cut_picture_x*2,Picture_max_y-Cut_picture_y*2);
    cv::Mat cut_dst_img = dst_img(cut_dst_img_rect);
    result_point = template_match(cut_dst_img,src_img);
    cut_src.x = result_point.x;
    cut_src.y = result_point.y;
    //cout << "Result point =" << result_point << " [template match]" << endl;
    src_point = result_point;
    unsigned short a = abs(result_point.x - Cut_picture_x);
    unsigned short b = abs(result_point.y - Cut_picture_y);


    //cout << " " << a << " " << b << " " << endl;
    if(a > Cut_picture_x/2 || b > Cut_picture_y/2) 
    {
        cout << "Serious misplacement of pictures" << endl;
        return 1;
    }
/*
    3. 利用模板匹配算法，计算图片大致错位情况，裁剪出较为相似图片 [cut_src_img,cut_dst_img]
*/

    cv::Rect cut_src_img_rect(result_point.x,result_point.y,Picture_max_x-Cut_picture_x*2,Picture_max_y-Cut_picture_y*2);
    cv::Mat cut_src_img = src_img(cut_src_img_rect);


/*
    4.1 利用金字塔模型和模板匹配算法，依次将相似图片裁剪
    4.2 利用模板匹配算法进行比较，若裁剪位置与检测位置差距过大，则裁剪图片为变化区域
    4.3 利用SSIM算法，通过亮度、对比度、结构等方面对裁剪图片和检测图片区域进行对比，判断图片相似度
*/  
    
    int cut_num =  Partition_num;
    // 裁剪大小设置 裁剪 8*8
    for(int i=0; i<cut_num; i++)
    {
        for(int j=0; j<cut_num; j++)
        {
            cv::Mat test_img = src_img;
        // 裁剪 检测区域
            dst.point_x = i*(Picture_max_x-Cut_picture_x*2)/cut_num ;
            dst.point_y = j*(Picture_max_y-Cut_picture_y*2)/cut_num ; 
            dst.width = (Picture_max_x-Cut_picture_x*2)/cut_num ;
            dst.height = (Picture_max_y-Cut_picture_y*2)/cut_num ;
            // 初步判断错位矫正后，裁剪小区域图片，进行匹配
            cv::Rect dst_rect(dst.point_x, dst.point_y, dst.width, dst.height);
            cv::Mat dst_image = cut_dst_img(dst_rect);
            
        // 裁剪 对比区域
            src.point_x = dst.point_x + cut_src.x - Cut_picture_x/2 ;
            src.point_y = dst.point_y + cut_src.y - Cut_picture_y/2 ;
            src.width = dst.width + Cut_picture_x ;
            src.height = dst.height + Cut_picture_y ;
            
            if(src.point_x + src.width > src_img.cols){
                src.width = src_img.cols - src.point_x-1;
            }
            if(src.point_y + src.height > src_img.rows){
                src.height = src_img.rows - src.point_y-1;
            }
            cv::Rect src_img_rect(src.point_x, src.point_y, src.width, src.height);
            cv::Mat  src_image_test = test_img(src_img_rect);

// 初步模板匹配，判断待测图片是否在原图片中存在
            result_point = template_match(dst_image,src_image_test);

            // 未找到
            if(result_point.x == 3000 && result_point.y == 3000)
            {
// 1.待测图片在原图中模板匹配不存在，比较待测图片与原图相同位置的图片相似度，相似度高，待测图在原图中存在，否则得出结论，不存在此图，框出目标   
                cv::Rect src_rect_test(cut_src.x + dst.point_x , cut_src.y + dst.point_y , dst.width, dst.height);
                
                //cout << "(i,j)=" << i << "," << j << endl;
                cv::Mat dst_image_temp = dst_image.clone();
                cv::Mat src_image = test_img(src_rect_test);
                cv::Mat src_image_temp = src_image.clone();               
            // SSIM
                blur(dst_image_temp,dst_image_temp,Size(10,10));
                blur(src_image_temp,src_image_temp,Size(10,10));
                Scalar SSIM1 = getMSSIM(dst_image_temp, src_image_temp);
                result_ssim = (SSIM1.val[2] + SSIM1.val[1] + SSIM1.val[0])/3 * 100;
                if(result_ssim > 90)
                {
            // 待测图片与原图相同位置图片，相似度高，得出结论，存在此图 
                    continue;
                }   
                float result1 = 100 - (float)ahash(dst_image_temp,src_image_temp)*100/256;
                if(result1 > 90){
                    continue;
                }
            // 待测图片与原图相同位置图片，相似度低，得出结论，不存在此图，框出目标 
                // 蓝色
                cv::rectangle(result_img, cv::Point((dst.point_x+Cut_picture_x)*Zoom_factor_x,(dst.point_y+Cut_picture_y)*Zoom_factor_y), cv::Point((dst.point_x + dst.width+Cut_picture_x)*Zoom_factor_x, (dst.point_y + dst.height+Cut_picture_y)*Zoom_factor_y), cv::Scalar(255,0,0), 8, 8); 
                pic_diff_num++;  
       
            }
            else
            {
            // 与错位图片比较
                cv::Rect src_rect_test(result_point.x, result_point.y, dst.width, dst.height);
                cv::Mat dst_image_temp = dst_image.clone();
                cv::Mat src_image = src_image_test(src_rect_test);
                cv::Mat src_image_temp = src_image.clone();               
                // SSIM
                blur(dst_image_temp,dst_image_temp,Size(10,10));
                blur(src_image_temp,src_image_temp,Size(10,10));
                Scalar SSIM1 = getMSSIM(dst_image_temp, src_image_temp);
                float result_ssim1 = (SSIM1.val[2] + SSIM1.val[1] + SSIM1.val[0])/3 * 100;
                if(result_ssim1 > 85){
                    continue;
                }
                float result1 = 100 - (float)ahash(dst_image_temp,src_image_temp)*100/256;
                if(result1 > 90){
                    continue;
                }
            // 与原图比较
                dst_image_temp = dst_image.clone();
                cv::Rect dst_rect_test(cut_src.x + dst.point_x, cut_src.y + dst.point_y, dst.width, dst.height);
                Mat src_image2 = test_img(dst_rect_test);
                src_image_temp = src_image2.clone();               
            // SSIM
                blur(dst_image_temp,dst_image_temp,Size(10,10));
                blur(src_image_temp,src_image_temp,Size(10,10));
                SSIM1 = getMSSIM(dst_image_temp, src_image_temp);
                float result_ssim2 = (SSIM1.val[2] + SSIM1.val[1] + SSIM1.val[0])/3 * 100;
                if(result_ssim2 > 85)
                {
                    continue;
                } 
                float result2 = 100 - (float)ahash(dst_image_temp,src_image_temp)*100/256;
                if(result2 > 90){
                    continue;
                }
/**********多种其他方法比较************/


// 3.模板匹配某些算法最佳位置相同，得出最佳相似起始坐标，判断与待测图片起始坐标是否相似，偏差较小，得出结论，存在此图
                if(result_point.x <= Cut_picture_x/2 + Location_differences && Cut_picture_x/2 <= result_point.x + Location_differences
                 &&result_point.y <= Cut_picture_y/2 + Location_differences && Cut_picture_y/2 <= result_point.y + Location_differences)
                {
                    // 淡蓝色 
                    cv::rectangle(result_img, cv::Point((dst.point_x+Cut_picture_x)*Zoom_factor_x,(dst.point_y+Cut_picture_y)*Zoom_factor_y), cv::Point((dst.point_x + dst.width+Cut_picture_x)*Zoom_factor_x, (dst.point_y + dst.height+Cut_picture_y)*Zoom_factor_y), cv::Scalar(255,255,0), 8, 8);
                    pic_diff_num++;
   
                }
// 4.模板匹配某些算法最佳位置相同，得出最佳相似起始坐标，判断与待测图片起始坐标是否相似，偏差中等，判断原图偏差位置图片与待测图片是否相似，相似则存在此图，否则不存在
                else if(result_point.x <= Cut_picture_x/2 + dst.width/4 && Cut_picture_x/2 <= result_point.x + dst.width/4
                 &&result_point.y <= Cut_picture_y/2 + dst.height/4 && Cut_picture_y/2 <= result_point.y + dst.height/4)
                {
                    // 绿色 
                    cv::rectangle(result_img, cv::Point((dst.point_x+Cut_picture_x)*Zoom_factor_x,(dst.point_y+Cut_picture_y)*Zoom_factor_y), cv::Point((dst.point_x + dst.width+Cut_picture_x)*Zoom_factor_x, (dst.point_y + dst.height+Cut_picture_y)*Zoom_factor_y), cv::Scalar(0,255,0), 8, 8);
                    pic_diff_num++;

                }
// 6.模板匹配某些算法最佳位置相同，得出最佳相似起始坐标，判断与待测图片起始坐标是否相似，偏差较大，取原图片相同位置比较
                else
                {
                    // 红色
                    cv::rectangle(result_img, cv::Point((dst.point_x+Cut_picture_x)*Zoom_factor_x,(dst.point_y+Cut_picture_y)*Zoom_factor_y), cv::Point((dst.point_x + dst.width+Cut_picture_x)*Zoom_factor_x, (dst.point_y + dst.height+Cut_picture_y)*Zoom_factor_y), cv::Scalar(0,0,255), 8, 8);
                    pic_diff_num++;

                }   
            }
        }
    }
    Det.template_Result = 100 - (float)(pic_diff_num*100*100  / (cut_num * cut_num)) /100 ;
    char updata = char(Det.template_Result);
    return 0;
}