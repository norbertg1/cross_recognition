/*****************************************************************************************************************************************/
/*	TO DO LIST:  Struktúrába írjam bele, a kereszt még hiányzó paramétereit, pl. mekkora a szög a kereszt hajalatainálgg
/*
/*****************************************************************************************************************************************/
#if __PLATFORM_WIN_ZERO_STANDARD__

#define _CRT_SECURE_NO_WARNINGS
#define snprintf sprintf_s 
typedef char uint8_t;

#endif

#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <string>
#include <vector>

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

/********************************************** Camera undistortion options **************************************************************/
//#define	intrinsic_path	"E:/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__intrinsic.xml"
#define		intrinsic_path	"../mobius_calibration_data/get_caloutput_20160413-2__intrinsic.xml"
//#define	distortion_path	"E:/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__distortion.xml"
#define		distortion_path	"../mobius_calibration_data/get_caloutput_20160413-2__distortion.xml"
/*****************************************************************************************************************************************/

/********************************************** General parameters ***********************************************************************/
#define	LINE_MAX_ANGLE						20					//In degrees
#define LINE_MIN_LENGHT						5					//In pixels
#define MIN_DEFECTS_SIZE					4					//see defects_size.png
#define MAX_DEFECTS_SIZE					20					//see defects_size.png
#define LINE_LENGHT_VARIANCE_PERCENT				35							//In percent
#define LINE_LENGHT_VARIANCE			LINE_LENGHT_VARIANCE_PERCENT/100
#define	IMSHOW												//Show captured pictures
#define IMSHOW_ERODEDILATE
#define crop_on
//#define crop_off
/*****************************************************************************************************************************************/
	
using namespace cv;
using namespace std;

struct Cross
{
	int		lenght[8];
	Point	candidate[4];	//A kereszt belső szögeinek a pontja
	Point	cross;			//Ezt keresem
	int		test1;			//kereszt belsejében levő derékszög tesztje
	int		test2;			//A vonalak megfelelő hosszuáságának tesztje
	int		test3;
};


void undistortion_code(cv::Mat *img);
void convexhulldefect(cv::Mat &img,vector<vector<Point> > &contours, std::vector< std::vector<Vec4i> > &defects);
void detect_cross(vector<vector<Point> > *contours,	std::vector< std::vector<Vec4i> > *defects, Cross *cross);
void detect_cross(vector<vector<Point> > &contours,	std::vector< std::vector<Vec4i> > &defects, Cross *cross);
void draw(vector<vector<Point> > &contours,	std::vector< std::vector<Vec4i> > &defects, Cross *cross, int i);
void draw1(vector<vector<Point> > &contours);
void erode_dilate(Mat &img,int dilation_size, int dilation_type, int iteration_dilate, int erode_size, int erode_type,int iteration_erode);
void ColorFilter(Mat &img);

Mat drawing,drawing1,img;

int main( int argc, char** argv )
{
	struct Cross cross;
	VideoCapture cap(1);
	//	cap.set(CV_CAP_PROP_FRAME_WIDTH,320);	cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
	int save_sequence=0;

	while(1){
		bool success = cap.read(img);
		if(!success){
			//img = imread("E:/object_recognition_c++/build/saved images/webcam_capture_0.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file, comment if use camera
			img = imread("saved images/webcam_capture_0.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file, comment if use camera
		}
		Mat img_orginal=img;
		undistortion_code(&img);	//Undistortion
//erode, dilate,
		vector<vector<Point> > contours;
		std::vector< std::vector<Vec4i> > defects;
		convexhulldefect( img, contours, defects);
		detect_cross(contours, defects, &cross );
#ifdef	IMSHOW
	//	imshow( "Orignal_image", img_orginal );
		imshow( "Undistorted_image", img );
		imshow( "drawing", drawing );
		imshow( "drawing1", drawing1 );
#endif
		imshow( "Undistorted_image", img );
		char c=cvWaitKey(10);		//press 'a' button for save picture
		if(c == 'c')	{printf("EXIT\n");	break;}
		if(c == 'a')	{char filename[25]; snprintf(filename,24,"webcam_capture_%d.jpg",save_sequence); printf("image saved"); imwrite(filename, img_orginal); save_sequence++;}
	}
	return 0;
}

/*************************************** Undistortion code ***********************************************************************/
/*  Need to define intrinsic_path, distortion_path
/*
/*	Example:
/*	#define intrinsic_path	"/home/username/Asztal/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__intrinsic.xml"
/*	#define	distortion_path	"/home/username/Asztal/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__distortion.xml"/*
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
/********************************** Convex hull and Convex defect*******************************************************************************************/
/*
/*	This function calculates convex hull and convex defect lines of image
/***********************************************************************************************************************************************************/
void convexhulldefect(cv::Mat &img,vector<vector<Point> > &contours, std::vector< std::vector<Vec4i> > &defects){
	Mat src_gray,threshold_output,dilate_output,erose_output,imgThresholded;
	erode_dilate(img,1,0,1,2,0,1);
	ColorFilter(img);	
	cvtColor( img, src_gray, CV_BGR2GRAY );
	//erode_dilate(src_gray,1,0,1,4,0,2);	
	blur( src_gray, src_gray, Size(3,3) );
	adaptiveThreshold(src_gray, threshold_output,  255 ,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,11,2);
//	threshold( src_gray, threshold_output, 85, 255, THRESH_BINARY );	/// Detect edges using Threshold
	imshow("threshold_output", threshold_output );
	erode_dilate(threshold_output,1,0,1,3,0,1);
	vector<Vec4i> hierarchy;
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	vector<vector<int> >hull_int( contours.size() );

	defects.resize( contours.size() );
	for( int i = 0; i < contours.size(); i++ )	{  
		convexHull( Mat(contours[i]), hull_int[i], false );
		if (contours[i].size() >3 )		convexityDefects(contours[i], hull_int[i], defects[i]);
	}
#ifdef IMSHOW	
	draw1(contours);
#endif
}
/********************************** Erode and Dilate **********************************************************************************************************/
/*
/*	Input: Input image as reference, dilation size, dilation type, number of iteration dilate, erode size, erode type, number of iteration erode
/*	
/*	#define IMSHOW ----->> for showing modified picture on the screen
/*a
/**************************************************************************************************************************************************************/
void erode_dilate(Mat &img,int dilation_size, int dilation_type, int iteration_dilate, int erode_size, int erode_type,int iteration_erode){

	Mat element = getStructuringElement( dilation_type,Size( 2*dilation_size + 1, 2*dilation_size+1 ),Point( dilation_size, dilation_size ) );
#ifdef IMSHOW_ERODEDILATE
	static uint8_t offset=0;
	char window_name[15];
	for(int i=0;i<iteration_dilate;i++){
		dilate(img,img,element);
		snprintf(window_name,12,"dilate %d",offset + i + 1);
		imshow(window_name, img );
	}
	element = getStructuringElement( erode_type,Size( 2*erode_size + 1, 2*erode_size+1 ),Point( erode_size, erode_size ) );	
	for(int i=0;i<iteration_erode;i++){
		erode(img,img,element);
		snprintf(window_name,12,"erode %d",offset + i + 1);	
		imshow(window_name, img );
	}
	offset += 10;
	if(offset>10) offset = 0;
#else
	for(int i=0;i<iteration_dilate;i++)	dilate(img,img,element);
	element = getStructuringElement( erode_type,Size( 2*erode_size + 1, 2*erode_size+1 ),Point( erode_size, erode_size ) );	
	for(int i=0;i<iteration_erode;i++)	erode(img,img,element);
#endif
}
/************************************************* Color filter ***************************************************************************************/
/*
/*	Eliminates picture part with inadequate color
/*
/******************************************************************************************************************************************************/

void ColorFilter(Mat &img){
	uint8_t dilation_size=7,dilation_type=0;
	Mat imgHSV,imgThresholded,imgThresholded1,img_masked;
	img_masked = Mat::zeros( img.size(), img.type() );
	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	inRange(imgHSV, Scalar(0, 120, 120), Scalar(10, 255, 255), imgThresholded);
	Mat element = getStructuringElement( dilation_type,Size( 2*dilation_size + 1, 2*dilation_size+1 ),Point( dilation_size, dilation_size ) );
	dilate(imgThresholded,imgThresholded,element);
	dilate(imgThresholded,imgThresholded,element);
	img.copyTo(img_masked, imgThresholded);
	img=img_masked;
	imshow("imgThresholded_dialted_mask", img_masked );
}
/**************************************************** Cross Detection algorithm *************************************************************************/
/*
/*
/*******************************************************************************************************************************************************/
void detect_cross(vector<vector<Point> > &contours,	std::vector< std::vector<Vec4i> > &defects, Cross *cross){
	printf("--------------------------------------------------contours.size(): %d --------------------------------------------------\n", contours.size());	
	for( int i = 0; i < contours.size(); i++ ){
	//	printf("-contour[i].size: %d\n",contours[i].size());
		if (defects[i].size() < MIN_DEFECTS_SIZE || defects[i].size() > MAX_DEFECTS_SIZE ) {/* printf("-Defects number FAILED\n");*/	continue; }	//defects.size() korlátozás
		cross->test1 = 0,cross->test2 = 0,cross->test3 = 0;
		//Végigmegyek az összes találl "defectsen" azt keresem ahol vagy pontosan négy derékszög(+-LINE_MAX_ANGLE) van és megvan a minimális hosszuk
		for(int j = 0; j<defects[i].size(); j++)  {
			Point ptStart( contours[i][defects[i][j][0]] );	//kék pontok
			Point ptEnd  ( contours[i][defects[i][j][1]] );	//piros pontok
			Point ptFar  ( contours[i][defects[i][j][2]] );	//zöld a kereszt belsejében levő pontok
			if(norm(ptFar-ptStart) < LINE_MIN_LENGHT)	{/*printf("--SHORT, line_lenght: %d FAILED\n", (int)norm(ptFar-ptStart));*/ continue;}	//Túl rövid vonalak kizárva
			float Angle1 = atan2(ptFar.y - ptStart.y, ptFar.x - ptStart.x) * 180.0 / CV_PI;
			float Angle2 = atan2(ptFar.y - ptEnd.y, ptFar.x - ptEnd.x) * 180.0 / CV_PI;
			if((abs(abs(Angle1)-abs(Angle2))-90)<LINE_MAX_ANGLE) {						//LINE_MAX_ANGLE_ELTERES kijavitani, mi azaz elteres?
				cross->candidate[cross->test1]=ptFar;										//A kereszt derékszögeit tesztelem 4 darabnak kell lennie
				cross->lenght[2*(cross->test1)]=norm(ptFar-ptStart);						//A kereszt szárainak a nagyságát töltöm ide
				cross->lenght[2*(cross->test1)+1]=norm(ptEnd-ptFar);
				cross->test1++;
				if(cross->test1>4) 	{printf("--TOO MUCH perpendicular>4 FAILED \n");	break;}	//Ha több mint 4 derékszög van az nem kereszt
			}
			printf("--Angle12: %.2f	Lenght: %d	cross.test1: %d\n",(float)(abs(abs(Angle1)-abs(Angle2))-90),(int)norm(ptFar-ptStart),cross->test1);
		}
		//A kereszt szárainak a hosszat tesztelem, az eltérés nem lehet nagyobb mint az átlaguk +- LINE_LENGHT_VARIANCE (százalékban)
		if(cross->test1 == 4){
			int line_lenght_avg1=0, line_lenght_avg2=0;
			for(int i=0;i<4;i++)	{line_lenght_avg1+=cross->lenght[2*i]; line_lenght_avg2+=cross->lenght[2*i+1];}
			line_lenght_avg1 = line_lenght_avg1/4;	line_lenght_avg2 = line_lenght_avg2/4;
			int line_lenght_tresh_max1=line_lenght_avg1+line_lenght_avg1*LINE_LENGHT_VARIANCE;
			int line_lenght_tresh_min1=line_lenght_avg1-line_lenght_avg1*LINE_LENGHT_VARIANCE;
			int line_lenght_tresh_max2=line_lenght_avg2+line_lenght_avg2*LINE_LENGHT_VARIANCE;
			int line_lenght_tresh_min2=line_lenght_avg2-line_lenght_avg2*LINE_LENGHT_VARIANCE;
			for(int i=0;i<4;i++){	
				if(cross->lenght[2*i]<line_lenght_tresh_max1 && cross->lenght[2*i]>line_lenght_tresh_min1 
					&& cross->lenght[2*i+1]<line_lenght_tresh_max2 && cross->lenght[2*i+1]>line_lenght_tresh_min2)	cross->test2++;	
			}	
		}
		//Ennek a működése elég kétséges, lehet elhagyható, A kereszt belső szögeit teszteli, viszont ha kereszt szárai elegendő hosszúságúak és a kereszt külső szögei megvannak 
		//akkor ez a feltétel is automatikusan teljesül
		if(cross->test2 == 4){
			float Angle[8];			
			Angle[0] = atan2(cross->candidate[0].y - cross->candidate[1].y, cross->candidate[0].x - cross->candidate[1].x) * 180.0 / CV_PI;
			Angle[1] = atan2(cross->candidate[1].y - cross->candidate[2].y, cross->candidate[1].x - cross->candidate[2].x) * 180.0 / CV_PI;

			Angle[2] = atan2(cross->candidate[1].y - cross->candidate[2].y, cross->candidate[1].x - cross->candidate[2].x) * 180.0 / CV_PI;
			Angle[3] = atan2(cross->candidate[2].y - cross->candidate[3].y, cross->candidate[2].x - cross->candidate[3].x) * 180.0 / CV_PI;

			Angle[4] = atan2(cross->candidate[2].y - cross->candidate[3].y, cross->candidate[2].x - cross->candidate[3].x) * 180.0 / CV_PI;
			Angle[5] = atan2(cross->candidate[3].y - cross->candidate[0].y, cross->candidate[3].x - cross->candidate[0].x) * 180.0 / CV_PI;

			Angle[6] = atan2(cross->candidate[3].y - cross->candidate[0].y, cross->candidate[3].x - cross->candidate[0].x) * 180.0 / CV_PI;
			Angle[7] = atan2(cross->candidate[0].y - cross->candidate[1].y, cross->candidate[0].x - cross->candidate[1].x) * 180.0 / CV_PI;

			for(int i=0;i<4;i++)	if((abs(abs(Angle[2*i])-abs(Angle[2*i+1]))-90)<LINE_MAX_ANGLE) {cross->test3++;}	

		/*	for(int i=0;i<4;i++)	printf("--Angle1: %.2f	Angel2: %.2f	Anlge12: %.2f	cross.test2: %d\n", Angle[2*i],Angle[2*i+1],
				(float)(abs(abs(Angle[2*i])-abs(Angle[2*i+1]))),cross->test2); */
		}								

		if(cross->test3==4){
			cross->cross.x=(cross->candidate[0].x+cross->candidate[1].x+cross->candidate[2].x+cross->candidate[3].x)/4;
			cross->cross.y=(cross->candidate[0].y+cross->candidate[1].y+cross->candidate[2].y+cross->candidate[3].y)/4;
			circle( drawing1,cross->cross,2,Scalar(0,0,255),5);
			circle( img, cross->cross,2,Scalar(255,0,0),5); 
		}
#ifdef IMSHOW	
		draw(contours, defects, cross, i );
#endif
	}
}
/********************************************** Drawing Visualize ************************************************************************************/
/*
/*
/*****************************************************************************************************************************************************/
void draw(vector<vector<Point> > &contours,	std::vector< std::vector<Vec4i> > &defects, Cross *cross, int i){
	printf("test1: %d	test2: %d	test3: %d",cross->test1,cross->test2,cross->test3);
	if(cross->test1 == 4){		//Ha megvan a négy belső derékszög
		printf("---TEST1 PASSED (Perpedicular lines)\n"); 					
		printf("---cros.x:\n");
		for(int i=0;i<4;i++) printf("---%d		cross_lenght: %d\n",cross->candidate[i].x,cross->lenght[2*i]);
		printf("---cros.y:\n");
		for(int i=0;i<4;i++) printf("---%d		cross_lenght: %d\n",cross->candidate[i].y,cross->lenght[2*i+1]);
		if(cross->test2 == 4){
			printf("----TEST2 PASSED (Lines lenght)\n"); 
			if(cross->test3 == 4){
				printf("----TEST3 PASSED (Perpedicular lines)\n"); 
			}
		}
	}

	for(int j = 0; j<defects[i].size(); j++)  {
		Point ptStart( contours[i][defects[i][j][0]] );	//kék pontok
		Point ptEnd  ( contours[i][defects[i][j][1]] );	//piros pontok
		Point ptFar  ( contours[i][defects[i][j][2]] );	//zöld a kereszt belsejében levő pontok

		line( drawing1, ptStart, ptEnd, Scalar(0,255,0), 1 );	//zöld
		line( drawing1, ptStart, ptFar, Scalar(255,255,0), 1 );	//sárga
		line( drawing1, ptEnd, ptFar, Scalar(0,255,255), 1 );	//kék
		circle( drawing1, ptFar, 4, Scalar(0,255,0), 1 );
		circle( drawing1, ptStart, 4, Scalar(255,0,0), 1 );
		circle( drawing1, ptEnd, 4, Scalar(0,0,255), 1 );
		char str[5];
		sprintf(str,"%d", j);
		//		putText(drawing1,str, Point(((ptStart.x+ptFar.x)/2),((ptStart.y+ptFar.y)/2)), FONT_HERSHEY_PLAIN, 1.0,CV_RGB(255,255,0), 1.0);
		//		putText(drawing1,str, Point(((ptEnd.x+ptFar.x)/2),((ptEnd.y+ptFar.y)/2)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 1.0);
		//		putText(drawing1,str, Point(ptFar.x+5,ptFar.y+5), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 1.0);
	}
}
void draw1(vector<vector<Point> > &contours){
	vector<vector<Point> >hull( contours.size() );
		for( int i = 0; i < contours.size(); i++ )	convexHull( Mat(contours[i]), hull[i], false );
		drawing = Mat::zeros( img.size(), CV_8UC3 );
		drawing1 = Mat::zeros( img.size(), CV_8UC3 );
		for( int i = 0; i< contours.size(); i++ ){
			drawContours( drawing, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );		//piros berajzolja a counturs-t
			drawContours( drawing, hull, i, Scalar(0,255,0), 1, 8, vector<Vec4i>(), 0, Point() );			//zöld	berajzolja a hull-t
		}
}




















void detect_cross(vector<vector<Point> > *contours,	std::vector< std::vector<Vec4i> > *defects, Cross *cross){
	printf("--------------------------------------------------contours.size(): %d --------------------------------------------------\n", contours->size());	
	for( int i = 0; i < contours->size(); i++ ){
		printf("-contour[i].size: %d\n",contours[i].size());
		if (defects[i].size() < MIN_DEFECTS_SIZE || defects[i].size() > MAX_DEFECTS_SIZE ) { printf("-Defects number FAILED\n");	continue; }	//defects.size() korlátozás
		cross->test1 = 0,cross->test2 = 0,cross->test3 = 0;
		//Végigmegyek az összes találl "defectsen" azt keresem ahol vagy pontosan négy derékszög(+-LINE_MAX_ANGLE) van és megvan a minimális hosszuk
		for(int j = 0; j<defects[i].size(); j++)  {
			Point ptStart( (*contours)[i][(*defects)[i][j][0]] );	//kék pontok
			Point ptEnd  ( (*contours)[i][(*defects)[i][j][1]] );	//piros pontok
			Point ptFar  ( (*contours)[i][(*defects)[i][j][2]] );	//zöld a kereszt belsejében levő pontok
			if(norm(ptFar-ptStart) < LINE_MIN_LENGHT)	{printf("--SHORT, line_lenght: %d FAILED\n", (int)norm(ptFar-ptStart)); continue;}	//Túl rövid vonalak kizárva
			float Angle1 = atan2(ptFar.y - ptStart.y, ptFar.x - ptStart.x) * 180.0 / CV_PI;
			float Angle2 = atan2(ptFar.y - ptEnd.y, ptFar.x - ptEnd.x) * 180.0 / CV_PI;
			if((abs(abs(Angle1)-abs(Angle2))-90)<LINE_MAX_ANGLE) {						//LINE_MAX_ANGLE_ELTERES kijavitani, mi azaz elteres?
				cross->candidate[cross->test1]=ptFar;										//A kereszt derékszögeit tesztelem 4 darabnak kell lennie
				cross->lenght[2*(cross->test1)]=norm(ptFar-ptStart);						//A kereszt szárainak a nagyságát töltöm ide
				cross->lenght[2*(cross->test1)+1]=norm(ptEnd-ptFar);
				cross->test1++;
				if(cross->test1>4) 	{printf("--TOO MUCH perpendicular>4 FAILED \n");	continue;}	//Ha több mint 4 derékszög van az nem kereszt
			}
			printf("--Angle12: %.2f	Lenght: %d	cross.test1: %d\n",(float)(abs(abs(Angle1)-abs(Angle2))-90),(int)norm(ptFar-ptStart),cross->test1);
		}
		//A kereszt szárainak a hosszat tesztelem, az eltérés nem lehet nagyobb mint az átlaguk +- LINE_LENGHT_VARIANCE (százalékban)
		if(cross->test1 == 4){
			int line_lenght_avg=0;
			for(int i=0;i<4;i++)	line_lenght_avg+=(cross->lenght[2*i]+cross->lenght[2*i+1]);
			line_lenght_avg = line_lenght_avg/8;
			int line_lenght_tresh_max=line_lenght_avg+line_lenght_avg*LINE_LENGHT_VARIANCE;
			int line_lenght_tresh_min=line_lenght_avg-line_lenght_avg*LINE_LENGHT_VARIANCE;
			for(int i=0;i<4;i++){	
				if(cross->lenght[2*i]<line_lenght_tresh_max && cross->lenght[2*i]>line_lenght_tresh_min 
					&& cross->lenght[2*i+1]<line_lenght_tresh_max && cross->lenght[2*i+1]>line_lenght_tresh_min)	cross->test2++;	
			}	
		}
		//Ennek a működése elég kétséges, lehet elhagyható, A kereszt belső szögeit teszteli, viszont ha kereszt szárai elegendő hosszúságúak és a kereszt külső szögei megvannak 
		//akkor ez a feltétel is automatikusan teljesül
		if(cross->test2 == 4){
			float Angle[8];			
			Angle[0] = atan2(cross->candidate[0].y - cross->candidate[1].y, cross->candidate[0].x - cross->candidate[1].x) * 180.0 / CV_PI;
			Angle[1] = atan2(cross->candidate[1].y - cross->candidate[2].y, cross->candidate[1].x - cross->candidate[2].x) * 180.0 / CV_PI;

			Angle[2] = atan2(cross->candidate[1].y - cross->candidate[2].y, cross->candidate[1].x - cross->candidate[2].x) * 180.0 / CV_PI;
			Angle[3] = atan2(cross->candidate[2].y - cross->candidate[3].y, cross->candidate[2].x - cross->candidate[3].x) * 180.0 / CV_PI;

			Angle[4] = atan2(cross->candidate[2].y - cross->candidate[3].y, cross->candidate[2].x - cross->candidate[3].x) * 180.0 / CV_PI;
			Angle[5] = atan2(cross->candidate[3].y - cross->candidate[0].y, cross->candidate[3].x - cross->candidate[0].x) * 180.0 / CV_PI;

			Angle[6] = atan2(cross->candidate[3].y - cross->candidate[0].y, cross->candidate[3].x - cross->candidate[0].x) * 180.0 / CV_PI;
			Angle[7] = atan2(cross->candidate[0].y - cross->candidate[1].y, cross->candidate[0].x - cross->candidate[1].x) * 180.0 / CV_PI;

			for(int i=0;i<4;i++)	if((abs(abs(Angle[2*i])-abs(Angle[2*i+1]))-90)<LINE_MAX_ANGLE) {cross->test3++;}	

			for(int i=0;i<4;i++)	printf("--Angle1: %.2f	Angel2: %.2f	Anlge12: %.2f	cross.test2: %d\n", Angle[2*i],Angle[2*i+1],
				(float)(abs(abs(Angle[2*i])-abs(Angle[2*i+1]))),cross->test2);
		}								

		if(cross->test3==4){
			cross->cross.x=(cross->candidate[0].x+cross->candidate[1].x+cross->candidate[2].x+cross->candidate[3].x)/4;
			cross->cross.y=(cross->candidate[0].y+cross->candidate[1].y+cross->candidate[2].y+cross->candidate[3].y)/4;
		}
//#ifdef IMSHOW	
		if(cross->test1 == 4){		//Ha megvan a négy belső derékszög
			printf("---TEST1 PASSED (Perpedicular lines)\n"); 					
			printf("---cros.x:\n");
			for(int i=0;i<4;i++) printf("---%d		cross_lenght: %d\n",cross->candidate[i].x,cross->lenght[2*i]);
			printf("---cros.y:\n");
			for(int i=0;i<4;i++) printf("---%d		cross_lenght: %d\n",cross->candidate[i].y,cross->lenght[2*i+1]);
			if(cross->test2 == 4){
				printf("----TEST2 PASSED (Lines lenght)\n"); 
				if(cross->test3 == 4){
					printf("----TEST3 PASSED (Perpedicular lines)\n"); 
				}
			}
		}

		for(int j = 0; j<defects[i].size(); j++)  {
			Point ptStart( (*contours)[i][(*defects)[i][j][0]] );	//kék pontok
			Point ptEnd  ( (*contours)[i][(*defects)[i][j][1]] );	//piros pontok
			Point ptFar  ( (*contours)[i][(*defects)[i][j][2]] );	//zöld a kereszt belsejében levő pontok
			//printf( "depth:%d   ",defects[i][j][3]);

			line( drawing1, ptStart, ptEnd, Scalar(0,255,0), 1 );	//zöld
			line( drawing1, ptStart, ptFar, Scalar(255,255,0), 1 );	//sárga
			line( drawing1, ptEnd, ptFar, Scalar(0,255,255), 1 );	//kék
			circle( drawing1, ptFar, 4, Scalar(0,255,0), 1 );
			circle( drawing1, ptStart, 4, Scalar(255,0,0), 1 );
			circle( drawing1, ptEnd, 4, Scalar(0,0,255), 1 );
			char str[5];
			sprintf(str,"%d", j);
			//		putText(drawing1,str, Point(((ptStart.x+ptFar.x)/2),((ptStart.y+ptFar.y)/2)), FONT_HERSHEY_PLAIN, 1.0,CV_RGB(255,255,0), 1.0);
			//		putText(drawing1,str, Point(((ptEnd.x+ptFar.x)/2),((ptEnd.y+ptFar.y)/2)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 1.0);
			//		putText(drawing1,str, Point(ptFar.x+5,ptFar.y+5), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 1.0);
		}
	}
//#endif
}
