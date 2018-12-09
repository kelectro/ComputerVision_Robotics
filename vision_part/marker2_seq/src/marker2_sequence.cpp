#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/tracking.hpp>
#include <dirent.h> // for linux systems
#include <sys/stat.h> // for linux systems
#include <vector>
#include <math.h>

#include "opencv2/opencv.hpp"

#include <chrono>  // for high_resolution_clock
# define M_PI           3.14159265358979323846 
using namespace cv;
using namespace std;


int readFilenames(vector<string> &filenames, const string &directory)
    {
        DIR *dir;
        class dirent *ent;
        class stat st;

        dir = opendir(directory.c_str());
        while ((ent = readdir(dir)) != NULL) {
            const string file_name = ent->d_name;
            const string full_file_name = directory + "/" + file_name;

            if (file_name[0] == '.')
                continue;

            if (stat(full_file_name.c_str(), &st) == -1)
                continue;

            const bool is_directory = (st.st_mode & S_IFDIR) != 0;

            if (is_directory)
                continue;

    //      filenames.push_back(full_file_name); // returns full path
            filenames.push_back(file_name); // returns just filename
        }
        closedir(dir);
           std::sort (filenames.begin(), filenames.end()); //optional, sort the filenames
        return(filenames.size()); //Return how many we found
    } // GetFilesInDirectory


void sobel(const cv::Mat& blurred)
{
    GaussianBlur(blurred, blurred, cv::Size(3, 3), 2, 0);
    // Derivative in the x direction
    cv::Mat grad_x;
    cv::Mat abs_grad_x;
    cv::Sobel(blurred, grad_x, CV_16S, 1, 0);
    cv::convertScaleAbs(grad_x, abs_grad_x); // Convert partial result to CV_8U
    cv::imshow("Derivative x", abs_grad_x);

    // Derivative in the y direction
    cv::Mat grad_y;
    cv::Mat abs_grad_y;
    cv::Sobel(blurred, grad_y, CV_16S, 0, 1);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::imshow("Derivative y", abs_grad_y);

    // Approximate gradient by adding both directional gradients
    cv::Mat grad;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    cv::imshow("Approximate gradient", grad);

    while (cv::waitKey() != 27)
        ;

    cv::destroyAllWindows();
}
void canny(Mat& img)
{

    Mat blured;
    Mat edges;
    Mat cdst;
    //cvtColor(img,blured,COLOR_BGR2GRAY);
    GaussianBlur(img, blured, cv::Size(3, 3), 2, 0);
    //cvtColor(blurred,cdst,COLOR_GRAY2BGR);
    Canny(blured, edges,180,240,3);
    imshow("Canny", edges);

    Mat dst=edges.clone();
    vector<Vec2f> lines;

    HoughLines(edges, lines, 1, CV_PI/180, 150, 50, 10 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(cdst, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
    }
    imshow("lines",cdst);


}


void marker2_easy(Mat& img)
{

    Mat src_gray;
    //(img,src_gray,COLOR_BGR2GRAY);
     //blur( img,img , Size(3,3) );

    imshow("original ",img);
    // canny(img);
     sobel(img);
}
int main( int argc, const char** argv )
    {
        //*****easy****
        string folder="/home/student/Desktop/rovi_final/vision_part/marker2_seq/sequence_2b/";
        //****hard***
        //string folder="/home/student/Desktop/rovi_final/marker_color/";
        cout << "Reading in directory " << folder << endl;
        vector<string> filenames;

        int num_files = readFilenames(filenames, folder);
        cout << "Number of files = " << num_files << endl;
         namedWindow( "image", 1 );
        for(size_t i = 0; i < filenames.size(); ++i)
        {
            cout << folder + filenames[i] << " #" << i << endl;
            Mat img =imread(folder + filenames[i]);
            if(!img.data) {//Protect against no file
                cerr << folder + filenames[i] << ", file #" << i << ", is not an image" << endl;
                continue;
            }

           marker2_easy(img);
           //marker2_hard(img);
          //cont(img);
         cv::waitKey(-1); //For fun, wait 250ms, or a quarter of a second, but you can put in "0" for no wait or -1 to wait for keypresses
           }
        }
