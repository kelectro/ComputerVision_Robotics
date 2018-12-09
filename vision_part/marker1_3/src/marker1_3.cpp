#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/tracking.hpp>
#include <vector>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>  // for high_resolution_clock
# define M_PI      3.14159265358979323846
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


vector<Point2f> cont(Mat&img)
{
            Mat canny_output;

            vector<std::vector<Point> > contours;
            vector<Vec4i> hierarchy;


            GaussianBlur(img, img, Size(9,9), 2,2);

            /// Find contours
            findContours( img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

            vector<Moments> mu(contours.size());
            for( int i = 0; i<contours.size(); i++ )
            { mu[i] = moments( contours[i], false ); }

            // get the centroid of figures.
            vector<Point2f> mc(contours.size());
            vector<Point2f>coordinates;
            for( int i = 0; i<contours.size(); i++)
            {
                mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );

            }
               for( int i = 0; i<contours.size(); i++ )
            {
                double perimeter = arcLength(contours[i], 1);
                double area = abs(contourArea(contours[i], true));

                if  ((4*M_PI*area)/(perimeter*perimeter)>0.75 && area>2000)

                {
                    cout<<mc[i];
                    coordinates.push_back(mc[i]);
   }
            }

return coordinates;
}

vector<Point2f> marker1_easy(Mat& img)
{
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    Mat red;
    Mat blue;
    Mat hsv;
    //convert to hsv color space
    cvtColor(img,hsv,COLOR_BGR2HSV);
    //separate blue and red circles in our image
    //thressholding using inrange function
    inRange(hsv, Scalar(0, 70, 50), Scalar(10,255,255), red);
    inRange(hsv, Scalar(111, 70, 50), Scalar(130,255,255), blue);
    // Erode+dilate to fill holes
    Mat kernel = Mat::ones(7, 7, CV_8U);
    dilate(blue, blue, kernel);
    erode(red, red, kernel);
    erode(blue,blue,kernel);
    erode(blue,blue,kernel);
   //combine both to a new binary img
    Mat out=red | blue;
    Mat original=img.clone();
    //blur to reduce noise
    GaussianBlur( out, red, Size(9, 9), 2, 2 );
    vector<Vec3f>circles;
    vector<Point2f>coordinates;
    /// Apply the Hough Transform to find the circles
    HoughCircles( red, circles, CV_HOUGH_GRADIENT, 1, red.rows/8, 80, 20, 0, 0 );

      /// Draw the circles detected
      for( size_t j = 0; j < circles.size(); j++ )
      {
          Point center(cvRound(circles[j][0]), cvRound(circles[j][1]));
          int radius = cvRound(circles[j][2]);
          coordinates.push_back(center);
          // circle center
         // circle( original, center, 3, Scalar(0,255,0), -1, 8, 0 );
          //circle( original, center, radius, Scalar(0,0,255), 3, 8, 0 );
          cout<<radius<<endl;
         }
       //Record end time
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        return coordinates;
}

vector<Point2f> marker1_hard(Mat& img)
{


    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    Mat red;
    Mat blue;
    Mat hsv;
    namedWindow("original",CV_WINDOW_NORMAL);
    imshow("original",img);

    //convert to hsv color space
    cvtColor(img,hsv,COLOR_BGR2HSV);
    //separate blue and red circles in our image
    //thressholding using inrange function
    inRange(hsv, Scalar(0, 70, 50), Scalar(10,255,255), red);
    inRange(hsv, Scalar(105, 30, 30), Scalar(120,255,255), blue);  //
    namedWindow("red",WINDOW_NORMAL);
    imshow("red",red);
    namedWindow("blue",WINDOW_NORMAL);
    imshow("blue",blue);
    // Erode+dilate to fill holes
   Mat kernel = Mat::ones(7, 7, CV_8U);
   dilate(blue, blue, kernel);
   erode(red, red, kernel);
   erode(blue,blue,kernel);
   erode(blue,blue,kernel);
   //imshow("Binary image", bin);
   Mat1b out=red | blue;
   vector<Point2f> coordinates;
   coordinates=cont(out);
   return coordinates;

}



// Extract the coordinates of the keypoints
vector<Point2f> keypoint_coordinates(const vector<KeyPoint>& keypoints)
{
    vector<Point2f> coords;
    coords.reserve(keypoints.size());

    for (const auto& kp : keypoints) {
        coords.push_back(kp.pt);    }

    return coords;
}


vector<Point2f>marker3(Mat& img1, Mat& img2)
{
    Ptr<Feature2D> detector;
    detector = cv::xfeatures2d::SURF::create();
    // Detect keypoints and compute descriptors
    vector<cv::KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    // Construct matcher
    Ptr<DescriptorMatcher> matcher;
    matcher = FlannBasedMatcher::create();
    // Find 2 nearest correspondences for each descriptor
    vector<vector<DMatch>> initial_matches;
    matcher->knnMatch(descriptors1, descriptors2, initial_matches, 2);
    cout << "Number of initial matches: " << initial_matches.size() << std::endl;

    // 1. Judge match quality based on Lowe's ratio criterion (from SIFT paper):
    // The ratio of the distance between the best and second-best match must be
    // less than 0.8
    vector<DMatch> matches;
    vector<KeyPoint> matched1;
    vector<KeyPoint> matched2;
    int idx = 0;

    for (const auto& match : initial_matches) {
        if (match[0].distance < 0.8f * match[1].distance) {
            matches.push_back(DMatch(idx, idx, match[0].distance));
            matched1.push_back(keypoints1[match[0].queryIdx]);
            matched2.push_back(keypoints2[match[0].trainIdx]);
            idx++;
        }
    }


    //2.Calculate perspective transformation
    vector<char> inlier_mask;
    Mat H =findHomography(keypoint_coordinates(matched1),
                                   keypoint_coordinates(matched2),
                                   cv::LMEDS,
                                   0,
                                   inlier_mask);


    int num_inliers = cv::countNonZero(inlier_mask);

    // Ratio of inliers vs original number of keypoint matches considered
    double ratio = double(num_inliers) / matches.size();;


    // Bounding box of original object (the book takes up the whole image area)
     vector<cv::Point2f> bb1{Point2f(0, 0),Point2f(img1.cols, 0),Point2f(img1.cols, img1.rows),Point2f(0, img1.rows)};
    // 3. Transform the bounding box points using the transformation matrix H
    vector<Point2f> bb2;
    perspectiveTransform(bb1, bb2, H);
    /*
    float coord[2];
    cout<<bb2[0]<<"Upper left corner"<<endl;
    cout<<bb2[1]<<"Upper right corner"<<endl;
    cout<<bb2[2]<<"Lower right corner"<<endl;
    cout<<bb2[3]<<"Lower left corner"<<endl;
    coord[0]=((bb2[1].x-bb2[0].x)/2);
    coord[1]=((bb2[3].y-bb2[1].y)/2);
    cout<<"Marker center X: "<<bb2[0].x+coord[0]<<"   "<<"Y: "<<bb2[0].y+coord[1]<<endl;
    */
    vector<Point2f>coord;
    for (int i=0;i<4;i++)
        coord.push_back(bb2[i]);

    return coord;



}

vector<Point2f>Vision(Mat& img,Mat img2,int flag)
{

    vector<Point2f>pointpairs;
    if (flag==0)
    pointpairs=marker1_easy(img);
    else if (flag==1)
    pointpairs=marker1_hard(img);
    else if (flag==2)
    pointpairs=marker3(img,img2);

    return pointpairs;
}

int main(int argc, char* argv[])
{
    ofstream out;
    out.open("runtime.txt",ios::app);
    //start runtime
    auto start = std::chrono::high_resolution_clock::now();

    vector<Point2f>coordinates;
    Mat img=imread("/home/student/Desktop/rovi_final/markers/Marker3.ppm");
    Mat img2=imread("/home/student/Desktop/rovi_final/vision_part/marker_corny/sequence_3/marker_corny_01.png");

    coordinates=Vision(img,img2,2);
    cout<<coordinates<<endl;
    //stop runtime
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    //Write runtime to file
    out << elapsed.count()<<endl;

}


