#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define SIGMA_ANTIALIAS			        0.5
#define SIGMA_PREBLUR			        1.0
#define INIT_SIGMA                      sqrt(2.0f)
#define OCTAVE_SMOOTHING_FACTOR         2
#define SCALE_SMOOTHING_FACTOR          sqrt(2.0f)


using namespace cv;
using namespace std;

const int NUM_SCALES = 5;
const int NUM_OCTAVES = 4;
const std::string IMG_LOC = "/Users/abhinandandubey/Desktop/cv/images/";


void display_adjacent(Mat im1, Mat im2){
    Mat newImage;
    hconcat(im1, im2, newImage);

    imshow("Display side by side", newImage);
    waitKey(0);
}

void stitch_octaves(Mat images[NUM_SCALES][NUM_OCTAVES]){
    for(int i=0; i< NUM_OCTAVES; i++){
        Mat octave_image;
        std::vector<cv::Mat> octave_stitch;
        for(int j=0; j< NUM_SCALES; j++){
            octave_stitch.push_back(images[j][i]);
        }
        vconcat(octave_stitch, octave_image);
        std::string image_title = IMG_LOC + "octave_" + std::to_string(i) + ".jpg";
        cout << image_title << endl;
        imwrite(image_title, octave_image);
    }

}


double **gaussian_kernel(int flavor=2, int W=5, double sigma=1.0){
    double** kernel = 0;
    kernel = new double*[W];
    double mean = W / 2;
    double sum1 = 0.0;
    double s = 2 * sigma * sigma;

    if(flavor == 1){
        double s = 2 * sigma * sigma * M_PI;
        for (int x = 0; x < W; ++x)
            for (int y = 0; y < W; ++y) {
                kernel[x][y] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
                               / (2 * M_PI * sigma * sigma);

                // Accumulate the kernel values
                sum1 += kernel[x][y];
            }

        // Normalize the kernel
        for (int x = 0; x < W; ++x)
            for (int y = 0; y < W; ++y)
                kernel[x][y] /= sum1;
    } else {
        for (int x = -W/2; x <= W/2; x++) {
            kernel[x+2] = new double[W];
            for (int y = -W / 2; y <= W / 2; y++) {
                kernel[x + 2][y + 2] = exp(-(x * x + y * y) / (s)) / (M_PI * s);
                sum1 += kernel[x + 2][y + 2];
            }
        }

        // Normalize the kernel
        for (int x = 0; x < W; ++x)
            for (int y = 0; y < W; ++y)
                kernel[x][y] /= sum1;
    }
    return kernel;
}


Mat convolute2D(bool if_debug, Mat image, double** kernel, int W){
    Mat filtered_image = image.clone();
    // find center position of kernel (half of kernel size)
    int kCenterX = W / 2;
    int kCenterY = W / 2;
    int xx = 0;
    int yy = 0;
    cout << endl << "Performing convolution .." << endl;
    cout << "Image Size : " << image.rows << ", " << image.cols <<endl;
    for (int i = 0; i < image.rows; ++i){
        for (int j = 0; j < image.cols; ++j){
            if(if_debug)
                cout << endl << "i, j = " << i << ", " << j << endl;
            for(int x = 0; x < W; ++x){
                xx = W - 1 - x;
                for(int y = 0; y < W; ++y){
                    yy = W - 1 - y;
                    if(if_debug)
                        cout << endl << "\t x, y = " << x << ", " << y;
                    int ii = i + (x - kCenterX);
                    int jj = j + (y - kCenterY);
                    if( ii >= 0 && ii < image.rows && jj >= 0 && jj < image.cols) {
                        if(if_debug){
                            cout << endl << "\t\t xx, yy = " << xx << ", " << yy;
                            cout << endl << "\t\t ii, jj = " << ii << ", " << jj;
                            cout << endl << "\t\t i, j = " << i << ", " << j << endl;
                            cout << "\t\t maxR, C = " << image.rows << ", " << image.cols <<endl;
                            cout << endl << "\t\t filtered_image.at<Vec2b>(Point(i, j)) " << (int)(filtered_image.at<uchar>(Point(j, i))) << ", image.at<Vec2b>(Point(ii, jj))" << (int)(image.at<uchar>(Point(jj, ii))) << ", kernel[xx][yy]" << kernel[xx][yy];
                        }
                        filtered_image.at<uchar>(Point(j, i)) += image.at<uchar>(Point(jj, ii)) * kernel[xx][yy];
                    }

                }
            }
        }
    }
    return filtered_image;
}


Mat gaussian_convolution(bool custom, bool visualize, Mat image, int W=5, double sigma=1.0) {
    if(custom){
        double **kernel = gaussian_kernel(2, W, sigma);
        std::ofstream out1("/Users/abhinandandubey/Desktop/cv/sift/gaussian.tsv");
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
            {
                out1 << double(i)/10 << '\t' << double(j)/10 << '\t' << kernel[i][j] << '\n';
                cout << kernel[i][j] << ' ';
            }
            out1 << '\n';
            cout << endl;
        }

        Mat filtered_image = convolute2D(false, image, kernel, W);
        if(visualize){
            display_adjacent(image, filtered_image);
        }
        return filtered_image;
    } else {
        Mat filtered_image_gaussian;
        GaussianBlur(image, filtered_image_gaussian, Size(W, W), 0, 0);
        if(visualize){
            display_adjacent(image, filtered_image_gaussian);
        }
        return filtered_image_gaussian;
    }
}



int main() {
    Mat orig_image, image, antialiased_image;
    Mat scale_space[NUM_SCALES][NUM_OCTAVES];
    double sigma_value[NUM_SCALES][NUM_OCTAVES];

    sigma_value[0][0] = INIT_SIGMA/2;
    for(int i = 0; i<NUM_OCTAVES; i++){
        if(i!=0) {
            sigma_value[0][i] = sigma_value[0][i-1] * OCTAVE_SMOOTHING_FACTOR;
        }
        for(int j = 1; j<NUM_SCALES; j++){
            sigma_value[j][i] = sigma_value[j-1][i] * SCALE_SMOOTHING_FACTOR;
        }
    }

    for(int i = 0; i<NUM_OCTAVES; i++){
        for(int j = 0; j<NUM_SCALES; j++){
            cout << sigma_value[j][i] << " | ";
        }
        cout << endl;
    }

    cout << "\n\n";

    try{
        orig_image = imread(IMG_LOC + "gray_citrus_crop.jpg", 0);
    } catch( cv::Exception& e ) {
        const char* err_msg = e.what();
        std::cout << "exception caught: " << err_msg << std::endl;
    }


    //scale_space[0][0] = orig_image;
    // Lowe claims blur the image with a sigma of 0.5 and double it's dimensions
    // to increase the number of stable keypoints
    cout << "Antialiasing the image a little ..." << endl;
    GaussianBlur(orig_image, antialiased_image, Size(5, 5), SIGMA_ANTIALIAS, SIGMA_ANTIALIAS);

    cout << "Upsampling the image to twice its size .." << endl;

    pyrUp(antialiased_image, scale_space[0][0], Size( antialiased_image.cols*2, antialiased_image.rows*2 ));

    cout << "Preblurring the base image .." << endl;

    // Preblur this base image
    GaussianBlur(scale_space[0][0], scale_space[0][0], Size(5, 5), SIGMA_PREBLUR, SIGMA_PREBLUR);

    for(int i = 1; i <= NUM_OCTAVES; i++){
        if(i < NUM_OCTAVES) {
            cout << "Octave : " << i-1 << endl;
            cout << "\t\tGenerating 0," << i << " by compressing 0," << i - 1 << endl;
            pyrDown(scale_space[0][i - 1], scale_space[0][i], Size(scale_space[0][i - 1].cols/2, scale_space[0][i - 1].rows/2 ) );
            //cout << ">>> scale_space[1][0].size() : " << scale_space[1][0].size() << endl;
        } else {
            cout << "Octave : " << i-1 << endl;
        }
        for(int j = 0; j < NUM_SCALES-1; j++){
            cout << "\tBlur Level : " << j << endl;
            scale_space[j+1][i-1] = gaussian_convolution(0, 0, scale_space[j][i-1]);
            cout << "\t\t Generating " << j+1 << "," << i-1 << " by blurring " << j << "," << i-1 << endl;
        }
    }

    //display_adjacent(scale_space[0][0], scale_space[2][0]);
    //display_adjacent(scale_space[0][1], scale_space[2][1]);
    stitch_octaves(scale_space); //octaves = across cols, num_cols, blurs = across rows, num_rows

    return 0;
}