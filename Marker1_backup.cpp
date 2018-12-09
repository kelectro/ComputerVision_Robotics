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

void circle_detect(Mat& img)
{
   Mat original=img.clone();
   GaussianBlur( img, img, Size(5, 5), 2, 2 );
   vector<Vec3f>circles;
   vector<Point>coordinates;
   vector<Vec3f>coordx;
   vector<Vec3f>coordy;
   vector<Vec3f>jacobi;
   /// Apply the Hough Transform to find the circles
   HoughCircles( img, circles, CV_HOUGH_GRADIENT, 1, img.rows/12, 80, 20, 45,61 );

   /// Draw the circles detected
   for( size_t j = 0; j < circles.size(); j++ )
   {
       Point center(cvRound(circles[j][0]), cvRound(circles[j][1]));
       int radius = cvRound(circles[j][2]);
       // circle center
       circle( original, center, 3, Scalar(0,255,0), -1, 8, 0 );
       cout<<center<<endl;
       coordinates.push_back(center);

       // circle outline
       circle( original, center, radius, Scalar(0,0,255), 3, 8, 0 );
       cout<<radius<<endl;

      }

     namedWindow( "Hough Circle Transform ", CV_WINDOW_AUTOSIZE );
   imshow( "Hough Circle Transform ", original );

}

void bound(Mat1b& mask, Mat&img)
{
    Mat img_bgr=img.clone();
    // Find non zero pixels
     vector<Point> pts;
     findNonZero(mask, pts);

     // Compute bounding box
     Rect box = boundingRect(pts);

     cout << "Width: " << box.width;
     cout << "Height: " << box.height << endl;

     // Show box
     rectangle(img_bgr, box, Scalar(0,0,255), 3);
     imshow("Box", img_bgr);
}


void cont(Mat&img)
{
        Mat image=img.clone();
        Mat imageGray=img.clone();
        Mat canny_output;

        //cvtColor( image, imageGray, CV_BGR2HSV );

             vector<std::vector<Point> > contours;
             vector<Vec4i> hierarchy;

            /// Detect edges using canny
            //int thresh =0;
             GaussianBlur(imageGray, image, Size(9,9), 2,2);
            //Canny( imageGray, canny_output, thresh, thresh*2, 3 );
            //imshow("canny",canny_output);
            //circle_detect(canny_output);


            /// Find contours
            findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

            vector<Moments> mu(contours.size());
            for( int i = 0; i<contours.size(); i++ )
            { mu[i] = moments( contours[i], false ); }

            // get the centroid of figures.
            vector<Point2f> mc(contours.size());
            for( int i = 0; i<contours.size(); i++)
            {
                mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );

            }


            // draw contours
            Mat drawing(img.size(), CV_8UC3, Scalar(255,255,255));
            for( int i = 0; i<contours.size(); i++ )
            {
                double perimeter = arcLength(contours[i], 1);
                double area = abs(contourArea(contours[i], true));

                if  ((4*M_PI*area)/(perimeter*perimeter)>0.75 && area>2000)

                {
                    cout<<mc[i];
            Scalar color = Scalar(167,151,0); // B G R values
            drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
            circle( drawing, mc[i], 4, color, -1, 8, 0 );
            //cout<<"perimeter="<<perimeter<<endl;
            //cout<<"Area="<<area<<endl;
            //cout<<"coordinates="<<mc[i]<<endl;
}
            }

            // show the resultant image
            namedWindow( "Contours", WINDOW_AUTOSIZE );
            imshow( "Contours", drawing );
            waitKey(0);
         /*

            // Start random number generator with a known seed
            RNG rng(12345);

            /// Draw contours
            Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
            for( int i = 0; i< contours.size(); i++ )
            {
                // Calculate perimeter length
                double perimeter = arcLength(contours[i], 1);
                double area = abs(contourArea(contours[i], true));
                double compactness = perimeter * perimeter / (4 * 3.141592 * area);
                printf("perimeter: %8.3f  area: %8.3f   compactness: %8.3f\n", perimeter, area, compactness);
                //circle_detect(drawing);
                if (area>2500)
                {
                    // Draw contour
                    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                    drawContours( drawing, contours, i, color, 4, 8, hierarchy, 0, Point() );


                }
               /* else
                {
                    // Draw contour
                    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                    drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
                }
            /*
            }



                //Show in a window
            namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
            imshow( "Contours", drawing );
*/


}



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

void marker1_easy(Mat& img)
{
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    Mat red;
    Mat blue;
    Mat hsv;
    Mat bin,bin2,blue1;
    namedWindow("original",CV_WINDOW_NORMAL);
    imshow("original",img);


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
   //imshow("Binary image", bin);
       Mat out=red | blue;
     imshow("out",out);
    //imshow("red",red);
    //imshow("blue  ",blue);
       Mat original=img.clone();

     GaussianBlur( out, red, Size(9, 9), 2, 2 );

      vector<Vec3f>circles;
      vector<Point>coordinates;
      vector<int>coordx;
      vector<int>coordy;
      vector<Vec3f>jacobi;
      /// Apply the Hough Transform to find the circles
      HoughCircles( red, circles, CV_HOUGH_GRADIENT, 1, red.rows/8, 80, 20, 0, 0 );

      /// Draw the circles detected
      for( size_t j = 0; j < circles.size(); j++ )
      {
          Point center(cvRound(circles[j][0]), cvRound(circles[j][1]));
          int radius = cvRound(circles[j][2]);
          // circle center
          circle( original, center, 3, Scalar(0,255,0), -1, 8, 0 );
          cout<<center<<endl;
          //coordinates.push_back(center);
          //coordx.push_back(center[j][0]);

          // circle outline
          circle( original, center, radius, Scalar(0,0,255), 3, 8, 0 );
          cout<<radius<<endl;


         }
       //Record end time
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      std::cout << "Elapsed time: " << elapsed.count() << " s\n";
      namedWindow( "Hough Circle Transform ", CV_WINDOW_AUTOSIZE );
      imshow( "Hough Circle Transform ", original );
}


void marker1_hard(Mat& img)
{


    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    Mat red;
    Mat blue;
    Mat hsv;
    Mat bin,bin2,blue1;
    namedWindow("original",CV_WINDOW_NORMAL);
    imshow("original",img);


    //convert to hsv color space
    cvtColor(img,hsv,COLOR_BGR2HSV);
    //separate blue and red circles in our image
    //thressholding using inrange function
    inRange(hsv, Scalar(0, 70, 50), Scalar(10,255,255), red);
    //inRange(hsv, Scalar(160, 70, 50), Scalar(180, 255, 255), bin);
    inRange(hsv, Scalar(105, 30, 30), Scalar(120,255,255), blue);  //110 70 50-->130.255.255
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
       cont(out);
      // bound(out,img);

    /*
     imshow("out",out);
    //imshow("red",red);
    //imshow("blue  ",blue);
       Mat original=img.clone();
     GaussianBlur( out, out, Size(9, 9), 2, 2 );
      Mat cont=out.clone();
      vector<Vec3f>circles;
      vector<Point>coordinates;
      vector<Vec3f>coordx;
      vector<Vec3f>coordy;
      vector<Vec3f>jacobi;
      /// Apply the Hough Transform to find the circles
      HoughCircles( out, circles, CV_HOUGH_GRADIENT, 1, out.rows/12, 80, 20, 45,61 );

      /// Draw the circles detected
      for( size_t j = 0; j < circles.size(); j++ )
      {
          Point center(cvRound(circles[j][0]), cvRound(circles[j][1]));
          int radius = cvRound(circles[j][2]);
          // circle center
          circle( original, center, 3, Scalar(0,255,0), -1, 8, 0 );
          cout<<center<<endl;
          coordinates.push_back(center);

          // circle outline
          circle( original, center, radius, Scalar(0,0,255), 3, 8, 0 );
          cout<<radius<<endl;

         }

        namedWindow( "Hough Circle Transform ", CV_WINDOW_AUTOSIZE );
      imshow( "Hough Circle Transform ", original );
      */
}
/*
void cont(Mat &img)

{
        Mat blue,red,hsv;

        //convert to hsv color space
        cvtColor(img,hsv,COLOR_BGR2HSV);
        //separate blue and red circles in our image
        //thressholding using inrange function
        inRange(hsv, Scalar(0, 70, 50), Scalar(10,255,255), red);
        //inRange(hsv, Scalar(160, 70, 50), Scalar(180, 255, 255), bin);
        inRange(hsv, Scalar(100, 70, 50), Scalar(130,255,255), blue);

        vector<vector<Point> > contours;
          vector<vector<Point> > circle_contours;
          vector<Vec4i> hierarchy;

          /// Find contours
          findContours( blue, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

          /* Extract correct contours
            for( int j = 0; j < contours.size(); j++ )
            {
              // Calculate parameters
              double area             = abs(contourArea(contours[j], true));
              double perimeter        = arcLength(contours[j], 1);
              double circle_constant  = (4 * PI * area) / (perimeter*perimeter);

              // Sort out bad contours
              if(perimeter > 100 && circle_constant > 0.7)
              {
                circle_contours.push_back(contours[j]);
              }
            }

        waitKey(-1);

}
        */
int main( int argc, const char** argv )
    {
        //*****easy****
        //string folder="/home/student/Desktop/rovi_final/marker_color/";
        //****hard***
        string folder="/home/student/Desktop/rovi_final/marker_color_hard/";
        cout << "Reading in directory " << folder << endl;
        vector<string> filenames;

        int num_files = readFilenames(filenames, folder);
        cout << "Number of files = " << num_files << endl;
         namedWindow( "image", 1 );
        for(size_t i = 0; i < filenames.size(); ++i)
        {
            cout << folder + filenames[i] << " #" << i << endl;
            Mat img =imread(folder + filenames[i]);

            if(!img.data) { //Protect against no file
                cerr << folder + filenames[i] << ", file #" << i << ", is not an image" << endl;
                continue;
            }
            // Blur image before edge detection

            //marker1_easy(img);

            marker1_hard(img);
          //cont(img);
              /*

                 for(int k=0;k<coordinates.capacity();k++)
                {
                     coordx.push_back(coordinates[k].y);
                     coordy.push_back(coordinates[k].y);
                     //cout<<"point x\t"<<coordinates[k].x<<endl;
                     cout<<"point y\t"<<coordinates[k].y<<endl;

                     //cout<<"y coord"<<coordy<<endl;
                     jacobi[k][0].push_back((coordx[3]-coordx[0])/2);
                }

                /*
                 for(int f=0;f<2;f++)
                 {
                    jacobi[f][0]=coordx[3]-coordx[0];
                    jacobi[f][1]=coordy[1]-coordx[0];

                 }
                 */
                //cout<<"vector"<<coordinates<<endl;
              /// Show your results


            cv::waitKey(200); //For fun, wait 250ms, or a quarter of a second, but you can put in "0" for no wait or -1 to wait for keypresses
            /* do whatever you want with your images here */
      }
        }

