#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

/********************************************** Camera undistortion options **************************************************************/
#define intrinsic_path	"/home/norbi/Asztal/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__intrinsic.xml"
#define	distortion_path	"/home/norbi/Asztal/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__distortion.xml"

#define crop_on
//#define crop_off
/*****************************************************************************************************************************************/

/********************************************** General parameters ***********************************************************************/
#define LINE_MAX_DISTANCE					50					//In pixels
#define LINE_MIN_DISTANCE					10					//In pixels
#define	LINE_MAX_ANGLE						10					//In degrees
#define LINE_MIN_LENGHT						20					//In pixels
#define MIN_DEFECTS_SIZE					4					//see defects_size.png
#define MAX_DEFECTS_SIZE					15					//see defects_size.png
#define LINE_LENGHT_VARIANCE_PERCENT				20					//In percent
#define MIN_CONTOURS_LENGHT					50					//In pixels nem biztos
#define LINE_LENGHT_VARIANCE			LINE_LENGHT_VARIANCE_PERCENT/100
#define	IMSHOW													//Show captured pictures
/*****************************************************************************************************************************************/

using namespace cv;

void readme();
void undistortion_code(cv::Mat *img);

int main( int argc, char** argv )
{
	uint16_t save_sequence=0;
	cv::Mat img;
	VideoCapture cap(1);
//	cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);

	while(1){
		bool bSuccess = cap.read(img);
		img = imread("/home/norbi/Asztal/object_recognition_c++/build/saved images/webcam_capture_2.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file, comment if use camera
		Mat img_orginal=img;
		undistortion_code(&img);	//Undistortion
/************************************** Convex hull and Convex defect algorithms *******************************************************/
//adaptive treshold, erode, dilate,
		Mat src_gray,threshold_output;
		vector<vector<Point> > contours;

	   	cvtColor( img, src_gray, CV_BGR2GRAY );
	   	blur( src_gray, src_gray, Size(3,3) );
		int thresh = 85;		
		threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );	/// Detect edges using Threshold
		imshow("threshold_output", threshold_output );	
		vector<Vec4i> hierarchy;
		findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		vector<vector<int> >hull_int( contours.size() );
		std::vector< std::vector<Vec4i> > defects( contours.size() );
		for( int i = 0; i < contours.size(); i++ )	{  
			convexHull( Mat(contours[i]), hull_int[i], false );
			if (contours[i].size() >3 )		convexityDefects(contours[i], hull_int[i], defects[i]);
		}
/************************************** Drawing ************************************************************************/		
		vector<vector<Point> >hull( contours.size() );
		for( int i = 0; i < contours.size(); i++ )	convexHull( Mat(contours[i]), hull[i], false );
		Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
		Mat drawing1 = Mat::zeros( threshold_output.size(), CV_8UC3 );
		for( int i = 0; i< contours.size(); i++ ){
			drawContours( drawing, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );		//piros berajzolja a counturs-t
			drawContours( drawing, hull, i, Scalar(0,255,0), 1, 8, vector<Vec4i>(), 0, Point() );			//zöld	berajzolja a hull-t
		}
/************************************** Detect algorithm ************************************************************************/
		printf("--------------------------------------------------contours.size(): %d --------------------------------------------------\n", contours.size());	
		for( int i = 0; i < contours.size(); i++ ){
			printf("contour[i].size: %d",contours[i].size());
			if (defects[i].size() < MIN_DEFECTS_SIZE || defects[i].size() > MAX_DEFECTS_SIZE ) { printf("*\n");	continue; }	//defects.size() korlátozás
			printf("\n");														//Ideális keresztnél 4 darab van
			int perpendicular1=0,perpendicular2=0,test=0,cross_lenght[8],j=0;;
			Point point_candidate[4];

			vector<Vec4i>::iterator d=defects[i].begin();	//d-t megnézni mit fejez ki
			for(int j = 0; j<defects[i].size(); j++)  {
				Point ptStart( contours[i][defects[i][j][0]] );
				Point ptEnd( contours[i][defects[i][j][1]] );
				Point ptFar( contours[i][defects[i][j][2]] );
				printf( "depth:%d   ",defects[i][j][3]);
			
				float Angle1, Angle2;
				Angle1 = atan2(ptFar.y - ptStart.y, ptFar.x - ptStart.x) * 180.0 / CV_PI;
				Angle2 = atan2(ptFar.y - ptEnd.y, ptFar.x - ptEnd.x) * 180.0 / CV_PI;
				if(norm(ptFar-ptStart) < LINE_MIN_LENGHT)	{printf("SHORT, line_lenght: %d\n", norm(ptFar-ptStart)); continue;}
				if((abs(abs(Angle1)-abs(Angle2))-90)<LINE_MAX_ANGLE) {
					point_candidate[perpendicular1]=ptFar;
					cross_lenght[2*perpendicular1]=norm(ptFar-ptStart);
					cross_lenght[2*perpendicular1+1]=norm(ptEnd-ptFar);
					perpendicular1++;
					if(perpendicular1>4)	{printf("TOO MUCH perpendicular>4");	continue;}
				}
			
				printf("Angle1: %.2f	Angle2: %.2f	", Angle1,Angle2);
				printf("Angle12: %.2f	Lenght: %d	",(float)(abs(abs(Angle1)-abs(Angle2))-90),norm(ptFar-ptStart));
				printf("Perpendicular lines: %d\n",perpendicular1);
			}	
			printf("------- i:%d	Perpendicular lines: %d ----------------------\n", i,perpendicular1);
			printf("\ncros.x:\n");
			for(int i=0;i<4;i++) printf("%d		cross_lenght: %d\n",point_candidate[i].x,cross_lenght[2*i]);
			printf("cros.y:\n");
			for(int i=0;i<4;i++) printf("%d		cross_lenght: %d\n",point_candidate[i].y,cross_lenght[2*i+1]);

			if(perpendicular1 == 4){
				float Angle[8];			//Ennek a működése elég kétséges, lehet elhagyható
				Angle[0] = atan2(point_candidate[0].y - point_candidate[1].y, point_candidate[0].x - point_candidate[1].x) * 180.0 / CV_PI;
				Angle[1] = atan2(point_candidate[1].y - point_candidate[2].y, point_candidate[1].x - point_candidate[2].x) * 180.0 / CV_PI;

				Angle[2] = atan2(point_candidate[1].y - point_candidate[2].y, point_candidate[1].x - point_candidate[2].x) * 180.0 / CV_PI;
				Angle[3] = atan2(point_candidate[2].y - point_candidate[3].y, point_candidate[2].x - point_candidate[3].x) * 180.0 / CV_PI;
				
				Angle[4] = atan2(point_candidate[2].y - point_candidate[3].y, point_candidate[2].x - point_candidate[3].x) * 180.0 / CV_PI;
				Angle[5] = atan2(point_candidate[3].y - point_candidate[0].y, point_candidate[3].x - point_candidate[0].x) * 180.0 / CV_PI;

				Angle[6] = atan2(point_candidate[3].y - point_candidate[0].y, point_candidate[3].x - point_candidate[0].x) * 180.0 / CV_PI;
				Angle[7] = atan2(point_candidate[0].y - point_candidate[1].y, point_candidate[0].x - point_candidate[1].x) * 180.0 / CV_PI;
				
				for(int i=0;i<4;i++)	if((abs(abs(Angle[2*i])-abs(Angle[2*i+1]))-90)<LINE_MAX_ANGLE) {perpendicular2++;}
				
				
				for(int i=0;i<4;i++)	printf("Angle1: %.2f	Angel2: %.2f	Anlge12: %.2f	perpendicular2: %d\n", Angle[2*i],Angle[2*i+1],
												(float)(abs(abs(Angle[2*i])-abs(Angle[2*i+1]))),perpendicular2);
				
				if(perpendicular2 == 4){
					int line_lenght_avg=0;
					for(int i=0;i<4;i++)	line_lenght_avg+=(cross_lenght[2*i]+cross_lenght[2*i+1]);
					line_lenght_avg = line_lenght_avg/8;
					int line_lenght_tresh_max=line_lenght_avg+line_lenght_avg*LINE_LENGHT_VARIANCE;
					int line_lenght_tresh_min=line_lenght_avg-line_lenght_avg*LINE_LENGHT_VARIANCE;
					for(int i=0;i<4;i++){	
						if(cross_lenght[2*i]<line_lenght_tresh_max && cross_lenght[2*i]>line_lenght_tresh_min 
							&& cross_lenght[2*i+1]<line_lenght_tresh_max && cross_lenght[2*i+1]>line_lenght_tresh_min)	test++;	
					}				
					printf("line_lenght_avg: %d		line_lenght_tresh_max: %d	line_lenght_tresh_min: %d",
							line_lenght_avg,line_lenght_tresh_max,line_lenght_tresh_min);				
					if(test==4){
						Point cross;				
						cross.x=(point_candidate[0].x+point_candidate[1].x+point_candidate[2].x+point_candidate[3].x)/4;
						cross.y=(point_candidate[0].y+point_candidate[1].y+point_candidate[2].y+point_candidate[3].y)/4;
						circle( drawing1, cross, 2, Scalar(0,0,255), 5 );
						circle( img, cross, 2, Scalar(255,0,0), 5 );
					}
				}
			}
			for(int j = 0; j<defects[i].size(); j++)  {
				Point ptStart( contours[i][defects[i][j][0]] );
				Point ptEnd( contours[i][defects[i][j][1]] );
				Point ptFar( contours[i][defects[i][j][2]] );
				printf( "depth:%d   ",defects[i][j][3]);

				line( drawing1, ptStart, ptEnd, Scalar(0,255,0), 1 );	//zöld
				line( drawing1, ptStart, ptFar, Scalar(255,255,0), 1 );	//sárga
				line( drawing1, ptEnd, ptFar, Scalar(0,255,255), 1 );	//kék
				circle( drawing1, ptFar, 4, Scalar(0,255,0), 1 );
				circle( drawing1, ptStart, 4, Scalar(255,0,0), 1 );
				circle( drawing1, ptEnd, 4, Scalar(0,0,255), 1 );
				char str[5];
				sprintf(str,"%d", j);
				printf("j: %d", j);
				putText(drawing1,str, Point(((ptStart.x+ptFar.x)/2),((ptStart.y+ptFar.y)/2)), FONT_HERSHEY_PLAIN, 1.0,CV_RGB(255,255,0), 1.0);
				putText(drawing1,str, Point(((ptEnd.x+ptFar.x)/2),((ptEnd.y+ptFar.y)/2)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 1.0);
				putText(drawing1,str, Point(ptFar.x+5,ptFar.y+5), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 1.0);
			}
		}
/********************************************************************************************************************************/	
#ifdef	IMSHOW
	//	imshow( "Orignal_image", img_orginal );
		imshow( "Undistorted_image", img );
		imshow( "drawing", drawing );
		imshow( "drawing1", drawing1 );
#endif
		char c=cvWaitKey(10);		//press 'a' button for save picture
		if(c == 'c')	{printf("EXIT\n");	break;}
		if(c == 'a')	{char filename[25]; snprintf(filename,24,"webcam_capture_%d.jpg",save_sequence); printf("image saved"); imwrite(filename, img_orginal); save_sequence++;}
	}
  return 0;
}

//void 
	
/*************************************** Undistortion code ***********************************************************************/
/*Need to define intrinsic_path, distortion_path
/*
/*	Example:
/*	#define intrinsic_path	"/home/norbi/Asztal/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__intrinsic.xml"
/*	#define	distortion_path	"/home/norbi/Asztal/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__distortion.xml"/*
/*
**********************************************************************************************************************************/
void undistortion_code(cv::Mat *img) {

	CvMat *intrinsic, *distortion;
	IplImage  *outputImg;
	IplImage inputImg = *img;		//convert Mat to IplImage	
									//	inputImg = &temp;
	outputImg = cvCreateImage(cvGetSize(&inputImg), inputImg.depth, 3);
	intrinsic = (CvMat*)cvLoad(intrinsic_path);
	distortion = (CvMat*)cvLoad(distortion_path);

#ifdef	crop_on
	cvUndistort2(&inputImg, outputImg, intrinsic, distortion);
	*img = cvarrToMat(outputImg);
#endif
#ifdef	crop_off
	double alpha = 1;
	CvMat *cameraMatrix = cvCreateMat(3, 3, CV_32FC1);
	IplImage *mapx = cvCreateImage(cvGetSize(inputImg), IPL_DEPTH_32F, 1);
	IplImage *mapy = cvCreateImage(cvGetSize(inputImg), IPL_DEPTH_32F, 1);
	cvGetOptimalNewCameraMatrix(intrinsic, distortion, cvGetSize(inputImg), alpha, cameraMatrix, cvGetSize(inputImg));
	cvInitUndistortRectifyMap(intrinsic, distortion, NULL, cameraMatrix, mapx, mapy);
	cvRemap(inputImg, outputImg, mapx, mapy);
#endif
	*img = cvarrToMat(outputImg);
}

/**********Opencv sedédlet példák stb..****************/
/*************************************** Convert image to gray, binary **********************************************************/
/*
	Mat image_temp,image_temp_gray;
	image_temp=img;
	cvtColor(image_temp,image_temp_gray,CV_RGB2GRAY);
	Mat img_binary = image_temp_gray > 64;			//Itt valtozik binarissa a kep
	img_binary = ~img_binary;				//negállom a képet

	image_temp=img_find;
	cvtColor(image_temp,image_temp_gray,CV_RGB2GRAY);
	Mat img_find_binary = image_temp_gray > 64;			//Itt valtozik binarissa a kep
	img_find_binary = ~img_find_binary;				//negállom a képet
*/
	
/********************************************************************************************************************************/
