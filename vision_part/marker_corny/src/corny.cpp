#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/tracking.hpp>
#include <dirent.h> // for linux systems
#include <sys/stat.h> // for linux systems
#include <vector>
#include <math.h>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>
#include <fstream>
#include "opencv2/opencv.hpp"

#include <chrono>  // for high_resolution_clock
# define M_PI           3.14159265358979323846 
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

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


// Extract the coordinates of the keypoints
std::vector<cv::Point2f> keypoint_coordinates(const std::vector<cv::KeyPoint>& keypoints)
{
    std::vector<cv::Point2f> coords;
    coords.reserve(keypoints.size());

    for (const auto& kp : keypoints) {
        coords.push_back(kp.pt);    }

    return coords;
}

// Draws bounding box defined by the points in 'bb'
void draw_bb(cv::Mat& img,
             const std::vector<cv::Point2f>& bb,
             const cv::Point2f& offset = cv::Point2f(0, 0))
{
    for (size_t i = 0; i < bb.size() - 1; i++) {
        cv::line(img, bb[i] + offset, bb[i + 1] + offset, cv::Scalar(0, 0, 255), 3);

    //cout<<"x1="<<bb[i]<<"\t"<<"y ="<<endl;
    }
    line(img, bb[bb.size() - 1] + offset, bb[0] + offset, cv::Scalar(0, 0, 255), 3);
    //cout<<"2"<<bb[bb.size() - 1]<<"22"<<bb[0]+offset<<endl;
}

int main( int argc, const char** argv )
    {
        string filepath1="/home/student/Desktop/rovi_final/vision_part/marker_corny/Marker3.ppm";
        Mat img1 = imread(filepath1);
        //*****easy****
        string folder="/home/student/Desktop/rovi_final/vision_part/marker_corny/sequence_3/";
        //****hard***
        //string folder="/home/student/Desktop/rovi_final/vision_part/marker_corny/sequence_3_h/";
        //cout << "Reading in directory " << folder << endl;
        vector<string> filenames;

        int num_files = readFilenames(filenames, folder);
        //cout << "Number of files = " << num_files << endl;
         namedWindow( "image", 1 );
        for(size_t i = 0; i < filenames.size(); ++i)
        {
            cout << folder + filenames[i] << " #" << i << endl;
            Mat img2 =imread(folder + filenames[i]);
            if(!img2.data) {//Protect against no file
                cerr << folder + filenames[i] << ", file #" << i << ", is not an image" << endl;
                continue;
            }

//*********************************start of detection ****************************************//
            //namedWindow("original",WINDOW_NORMAL);
            //imshow("original",img2);
              // Construct detector
                    ofstream out;
                    out.open("runtime.txt",ios::app);
                 // Record start time
                auto start = std::chrono::high_resolution_clock::now();
                Mat red;

                Ptr<cv::Feature2D> detector;
                detector = cv::xfeatures2d::SURF::create();



                // Detect keypoints and compute descriptors
                std::vector<cv::KeyPoint> keypoints1, keypoints2;
                cv::Mat descriptors1, descriptors2;
                detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
                detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

                // Construct matcher
                cv::Ptr<cv::DescriptorMatcher> matcher;




                 matcher = cv::FlannBasedMatcher::create();


                // Find 2 nearest correspondences for each descriptor
                std::vector<std::vector<cv::DMatch>> initial_matches;
                matcher->knnMatch(descriptors1, descriptors2, initial_matches, 2);
                //std::cout << "Number of initial matches: " << initial_matches.size() << std::endl;

                // 1. Judge match quality based on Lowe's ratio criterion (from SIFT paper):
                // The ratio of the distance between the best and second-best match must be
                // less than 0.8
                std::vector<cv::DMatch> matches;
                std::vector<cv::KeyPoint> matched1;
                std::vector<cv::KeyPoint> matched2;
                int idx = 0;

                for (const auto& match : initial_matches) {
                    if (match[0].distance < 0.8f * match[1].distance) {
                        matches.push_back(cv::DMatch(idx, idx, match[0].distance));
                        matched1.push_back(keypoints1[match[0].queryIdx]);
                        matched2.push_back(keypoints2[match[0].trainIdx]);
                        idx++;
                    }
                }

                // We only use the good matches
                auto num_matches = matches.size();

                //std::cout << "Number of good matches: " << num_matches << std::endl;

                if (num_matches < 4) {
                    std::cout << "Too few matches!" << std::endl;
                    return 0;
                }

                // 2. Calculate perspective transformation
                std::vector<char> inlier_mask;
                cv::Mat H = cv::findHomography(keypoint_coordinates(matched1),
                                               keypoint_coordinates(matched2),
                                               cv::LMEDS,
                                               0,
                                               inlier_mask);

                if (H.empty()) {
                    std::cout << "H matrix could not be estimated!" << std::endl;
                    return 0;
                }

                int num_inliers = cv::countNonZero(inlier_mask);
                //std::cout << "Number of inliers: " << num_inliers << std::endl;

                // Ratio of inliers vs original number of keypoint matches considered
                double ratio = double(num_inliers) / num_matches;
                //std::cout << "Inlier ratio: " << ratio << std::endl;

                // Bounding box of original object (the book takes up the whole image area)
                std::vector<cv::Point2f> bb1{
                    cv::Point2f(0, 0),
                    cv::Point2f(img1.cols, 0),
                    cv::Point2f(img1.cols, img1.rows),
                    cv::Point2f(0, img1.rows)
                };

                // 3. Transform the bounding box points using the transformation matrix H
                vector<Point2f> bb2;
                perspectiveTransform(bb1, bb2, H);
                float coord[2];
                // Visualize the result
                Mat img_out;
                drawMatches(img1, matched1,
                                img2, matched2,
                                matches, img_out,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                inlier_mask); // Draw only inliers
                draw_bb(img_out, bb1);
                draw_bb(img_out, bb2, cv::Point2f(img1.cols, 0)); // Offset by img1.cols in the x direction
                cv::imshow("Matches", img_out);

               /* cout<<bb2[0]<<"Upper left corner"<<endl;
                cout<<bb2[1]<<"Upper right corner"<<endl;
                cout<<bb2[2]<<"Lower right corner"<<endl;
                cout<<bb2[3]<<"Lower left corner"<<endl;
                coord[0]=((bb2[1].x-bb2[0].x)/2);
                coord[1]=((bb2[3].y-bb2[1].y)/2); */
                cout<<"Marker center X: "<<bb2[0].x+coord[0]<<"   "<<"Y: "<<bb2[0].y+coord[1]<<endl;
                //Record end time
               auto finish = std::chrono::high_resolution_clock::now();
               std::chrono::duration<double> elapsed = finish - start;
               std::cout << "Elapsed time: " << elapsed.count() << " s\n";
               out << elapsed.count()<<endl;
                out.close();

             cv::waitKey(200); //For fun, wait 250ms, or a quarter of a second, but you can put in "0" for no wait or -1 to wait for keypresses
           }
        }
