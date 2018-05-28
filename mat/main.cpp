#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <mach/mach.h>


using namespace cv;

struct task_basic_info t_info;
mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

void getMemUsage(){
    if (KERN_SUCCESS != task_info(mach_task_self(),
                                  TASK_BASIC_INFO, (task_info_t)&t_info,
                                  &t_info_count))
    {
        std::cout << "Memory Usage : -1" << std::endl;
    } else {
        std::cout << "Memory Usage : " << t_info.resident_size/1000000 << "MB. " << std::endl;
    }
}

int main(int argc, char** argv ) {
    std::cout << "Hello, World!" << std::endl;

    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    getMemUsage(); // Memory Usage : 14MB.

    try{
        image = imread( argv[1], 1 );
    } catch( cv::Exception& e ) {
        const char* err_msg = e.what();
        std::cout << "exception caught: " << err_msg << std::endl;
    }

    getMemUsage(); // Memory Usage : 27MB.

    Mat another_image = image;

    getMemUsage(); // Memory Usage : 27MB. No change in memory utilization.

    Mat cloned_image = image.clone(); //Explicitly asking OpenCV to copy the data itself.

    getMemUsage(); // Memory Usage : 39MB. A jump of 12MB

    waitKey(0);
    return 0;
}