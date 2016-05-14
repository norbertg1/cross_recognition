/*****************************************************************************************************************************************/
/*	TO DO LIST:  Struktúrába írjam bele, a kereszt még hiányzó paramétereit, pl. mekkora a szög a kereszt hajalatainálgg
/*
/*****************************************************************************************************************************************/
#if __PLATFORM_WIN_ZERO_STANDARD__

	#define _CRT_SECURE_NO_WARNINGS
	#define snprintf sprintf_s 
	typedef char uint8_t;

#endif

#ifdef __x86_64__
	#define	IMSHOW												//Show captured pictures
	//#define IMSHOW_ERODEDILATE
	//#define IMSHOW_DILATEERODE
	#define CAMERA 1
#else
	#define CAMERA 0
#endif

#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>

/********************************************** Camera undistortion options **************************************************************/
//#define	intrinsic_path	"E:/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__intrinsic.xml"
#define		intrinsic_path	"calibrate_images/__intrinsic.xml"
//#define	distortion_path	"E:/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__distortion.xml"
#define		distortion_path	"calibrate_images/__distortion.xml"
/*****************************************************************************************************************************************/

/********************************************** General parameters ***********************************************************************/
#define	LINE_MAX_ANGLE						20					//In degrees
#define LINE_MIN_LENGHT						5					//In pixels
#define MIN_DEFECTS_SIZE					4					//see defects_size.png
#define MAX_DEFECTS_SIZE					20					//see defects_size.png
#define LINE_LENGHT_VARIANCE_PERCENT				35					//In percent
#define LINE_LENGHT_VARIANCE			LINE_LENGHT_VARIANCE_PERCENT/100
#define crop_on
//#define crop_off
/*****************************************************************************************************************************************/

	
using namespace cv;
using namespace std;

struct Cross
{
	int		lenght[8];
	Point		candidate[4];	//A kereszt belső szögeinek a pontja
	Point		cross;			//Ezt keresem
	int		test1;			//kereszt belsejében levő derékszög tesztje
	int		test2;			//A vonalak megfelelő hosszuáságának tesztje
	int		test3;
	int		r;
};

void undistortion_code(cv::Mat *img);
void convexhulldefect(cv::Mat &img,vector<vector<Point> > &contours, std::vector< std::vector<Vec4i> > &defects);
void detect_cross(vector<vector<Point> > *contours,	std::vector< std::vector<Vec4i> > *defects, Cross *cross);
void detect_cross(vector<vector<Point> > &contours,	std::vector< std::vector<Vec4i> > &defects, Cross *cross);
void draw(vector<vector<Point> > &contours,	std::vector< std::vector<Vec4i> > &defects, Cross *cross, int i);
void draw1(vector<vector<Point> > &contours);
void dilate_erode(Mat &img, int dilation_size, int dilation_type, int iteration_dilate, int erode_size, int erode_type,int iteration_erode);
void erode_dilate(Mat &img, int erode_size, int erode_type,int iteration_erode, int dilation_size, int dilation_type, int iteration_dilate);
void ColorFilter(Mat &img);

Mat drawing,drawing1,img;
	struct Cross cross;
int main( int argc, char** argv )
{

	VideoCapture cap(CAMERA);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH,320);	cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
	int fps,FPS,save_sequence=0;
	clock_t seconds = clock();
	clock_t t[10];
	while(1){
		t[0]=clock();
		fps++;
		if(seconds != clock()/CLOCKS_PER_SEC)	{FPS=fps/(clock()/ CLOCKS_PER_SEC - seconds); printf(" FPS: %d\n", FPS); fps=0, seconds=clock()/ CLOCKS_PER_SEC;}
		bool success = cap.read(img);
		if(!success){
			//img = imread("E:/object_recognition_c++/build/saved images/webcam_capture_0.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file, comment if use camera
			img = imread("webcam_capture_1.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file, comment if use camera
		}
		t[1]=clock()-t[0];
		Mat img_orginal=img;
		undistortion_code(&img);	//Undistortion
		t[2]=clock()-t[1]-t[0];
		vector<vector<Point> > contours;
		std::vector< std::vector<Vec4i> > defects;
		convexhulldefect( img, contours, defects);
		t[3]=clock()-t[0]-t[1]-t[2];
		detect_cross(contours, defects, &cross );
		t[4]=clock()-t[0]-t[1]-t[2]-t[3];
#ifdef	IMSHOW
//		imshow( "Orignal_image", img_orginal );
		imshow( "Undistorted_image", img );
		imshow( "drawing", drawing );
		imshow( "drawing1", drawing1 );
		imshow( "Undistorted_image", img );
//		imshow( "Orignal_image", img_orginal );
#endif
		t[5]=clock()-t[0]-t[1]-t[2]-t[3]-t[4];
		char c=0;//cvWaitKey(10);		//press 'a' button for save picture
		if(c == 'c')	{printf("EXIT\n");	break;}
		if(c == 'a')	{char filename[25]; snprintf(filename,24,"webcam_capture_%d.jpg",save_sequence); printf("image saved"); imwrite(filename, img_orginal); save_sequence++;}
		t[6]=clock()-t[0]-t[1]-t[2]-t[3]-t[5];
		printf("t[0]: %d	t[1]: %d	t[2]: %d	t[3]: %d	t[4]: %d	t[5]: %d	FPS: %d CLOCKS_PER_SEC%d\n", (int)((t[0])), (int)(t[1]*1000/CLOCKS_PER_SEC), (int)(t[2]*1000/CLOCKS_PER_SEC), (int)(t[3]*1000/CLOCKS_PER_SEC), (int)(t[4]*1000/CLOCKS_PER_SEC), (int)(t[5]*1000/CLOCKS_PER_SEC), FPS, CLOCKS_PER_SEC);
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
/*
/***********************************************************************************************************************************************************/
void convexhulldefect(cv::Mat &img,vector<vector<Point> > &contours, std::vector< std::vector<Vec4i> > &defects){
	Mat src_gray,threshold_output,dilate_output,erose_output,imgThresholded;
	dilate_erode(img,1,0,1,2,0,1);
	
	ColorFilter(img);	
	cvtColor( img, src_gray, CV_BGR2GRAY );
//	blur( img, img, Size(7,7) );
//	imshow("blur", img );
	//dilate_erode(src_gray,1,0,1,4,0,2);	
	adaptiveThreshold(src_gray, threshold_output,  255 ,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,11,2);
//	threshold( src_gray, threshold_output, 85, 255, THRESH_BINARY );	/// Detect edges using Threshold
//	imshow("threshold_output", threshold_output );
	dilate_erode(threshold_output,1,0,1,1,0,1);
//	imshow("threshold_output", threshold_output );
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
/********************************** Dilate and Erode  **********************************************************************************************************/
/*
/*	Input: Input image as reference, dilation size, dilation type, number of iteration dilate, erode size, erode type, number of iteration erode
/*	
/*	#define IMSHOW_ERODEDILATE ----->> for showing modified picture on the screen
/*
/**************************************************************************************************************************************************************/
void dilate_erode(Mat &img,int dilation_size, int dilation_type, int iteration_dilate, int erode_size, int erode_type,int iteration_erode){

	Mat element = getStructuringElement( dilation_type,Size( 2*dilation_size + 1, 2*dilation_size+1 ),Point( dilation_size, dilation_size ) );
#ifdef IMSHOW_DILATEERODE
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
/********************************** Erode and Dilate **********************************************************************************************************/
/*
/*	Input: Input image as reference, dilation size, dilation type, number of iteration dilate, erode size, erode type, number of iteration erode
/*	
/*	#define IMSHOW_ERODEDILATE ----->> for showing modified picture on the screen
/*
/**************************************************************************************************************************************************************/
void erode_dilate(Mat &img, int erode_size, int erode_type,int iteration_erode, int dilation_size, int dilation_type, int iteration_dilate){

	Mat element = getStructuringElement( erode_type,Size( 2*erode_size + 1, 2*erode_size+1 ),Point( erode_size, erode_size ) );
#ifdef IMSHOW_ERODEDILATE
	static uint8_t offset=0;
	char window_name[15];
	for(int i=0;i<iteration_erode;i++){
		erode(img,img,element);
		snprintf(window_name,12," erode %d",offset + i + 1);	
		imshow(window_name, img );
	}
	element = getStructuringElement( dilation_type,Size( 2*dilation_size + 1, 2*dilation_size+1 ),Point( dilation_size, dilation_size ) );
	for(int i=0;i<iteration_dilate;i++){
		dilate(img,img,element);
		snprintf(window_name,12," dilate %d",offset + i + 1);
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
	Mat imgHSV,imgThresholded,imgThresholded1,imgThresholded2,img_masked,color_measurement_mask;
	img_masked = Mat::zeros( img.size(), img.type() );

	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	inRange(imgHSV, Scalar(0, 120, 90), Scalar(10, 255, 255), imgThresholded1);
	inRange(imgHSV, Scalar(170, 120, 90), Scalar(180, 255, 255), imgThresholded2);
//	erode_dilate(imgThresholded1,1,0,1,100,0,10);
//	erode_dilate(imgThresholded2,1,0,1,100,0,10);
//	imshow("imgThresholded11", imgThresholded1 );
//	imshow("imgThresholded22", imgThresholded2 );
	imgThresholded=Mat::zeros( imgThresholded1.size(), imgThresholded1.type() );
	bitwise_or(imgThresholded1,imgThresholded2,imgThresholded);
//	circle( imgThresholded,cross.cross,2*cross.r,255,-1);	
	img.copyTo(img_masked, imgThresholded);

//	imshow("imgThresholded22", imgThresholded );
	img=img_masked;
//	imshow("imgThresholded_dialted_mask", img_masked );
}
/**************************************************** Cross Detection algorithm *************************************************************************/
/*
/*
/*******************************************************************************************************************************************************/
void detect_cross(vector<vector<Point> > &contours,	std::vector< std::vector<Vec4i> > &defects, Cross *cross){
//	printf("--------------------------------------------------contours.size(): %d --------------------------------------------------\n", contours.size());	
	for( int i = 0; i < contours.size(); i++ ){
	//	printf("-contour[i].size: %d\n",contours[i].size());
		if (defects[i].size() < MIN_DEFECTS_SIZE || defects[i].size() > MAX_DEFECTS_SIZE ) { printf("-Defects number: %d FAILED\n", defects[i].size());	continue; }	//defects.size() korlátozás
		cross->test1 = 0,cross->test2 = 0,cross->test3 = 0;
		//Végigmegyek az összes találl "defectsen" azt keresem ahol vagy pontosan négy derékszög(+-LINE_MAX_ANGLE) van és megvan a minimális hosszuk
		for(int j = 0; j<defects[i].size(); j++)  {
			Point ptStart( contours[i][defects[i][j][0]] );	//kék pontok
			Point ptEnd  ( contours[i][defects[i][j][1]] );	//piros pontok
			Point ptFar  ( contours[i][defects[i][j][2]] );	//zöld a kereszt belsejében levő pontok
			if(norm(ptFar-ptStart) < LINE_MIN_LENGHT)	{/*printf("--SHORT, line_lenght: %d FAILED\n", (int)norm(ptFar-ptStart));*/ continue;}	//Túl rövid vonalak kizárva
			float Angle1 = atan2(ptFar.y - ptStart.y, ptFar.x - ptStart.x) * 180.0 / CV_PI;
			float Angle2 = atan2(ptFar.y - ptEnd.y, ptFar.x - ptEnd.x) * 180.0 / CV_PI;
			float Angle12, Angle21;			
			if(Angle1>Angle2)	Angle12 = Angle1 - Angle2;
			else				Angle12 = Angle2 - Angle1;
			Angle21 = Angle12 - 180;

			if(abs(Angle12-90)<LINE_MAX_ANGLE || abs(Angle21-90)<LINE_MAX_ANGLE) {						//LINE_MAX_ANGLE_ELTERES kijavitani, mi azaz elteres?
				cross->candidate[cross->test1]=ptFar;										//A kereszt derékszögeit tesztelem 4 darabnak kell lennie
				cross->lenght[2*(cross->test1)]=norm(ptFar-ptStart);						//A kereszt szárainak a nagyságát töltöm ide
				cross->lenght[2*(cross->test1)+1]=norm(ptEnd-ptFar);
				cross->test1++;
			//	if(cross->test1>4) 	{printf("--TOO MUCH perpendicular>4 FAILED \n");	break;}	//Ha több mint 4 derékszög van az nem kereszt
			}
//			printf("--Angle1: %.2f	Angle2: %.2f	Angle12: %.2f	Angle12: %.2f	Lenght: %d	cross.test1: %d\n",Angle1,Angle2,Angle12,Angle21,(int)norm(ptFar-ptStart),cross->test1);
		}
		//A kereszt szárainak a hosszat tesztelem, az eltérés nem lehet nagyobb mint az átlaguk +- LINE_LENGHT_VARIANCE (százalékban)
		if(cross->test1 == 4){
			int line_lenght_avg1=0, line_lenght_avg2=0;
			for(int i=0;i<4;i++)	{line_lenght_avg1+=cross->lenght[2*i]; line_lenght_avg2+=cross->lenght[2*i+1];}
			line_lenght_avg1 = line_lenght_avg1/4;	line_lenght_avg2 = line_lenght_avg2/4;
			cross->r=(line_lenght_avg1 + line_lenght_avg2)/2;
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
			printf(" Cross position: x: %d	y: %d\n", cross->cross.x, cross->cross.y); 
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
	printf("test1: %d	test2: %d	test3: %d\n",cross->test1,cross->test2,cross->test3);
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
/*		Mat conturs = Mat::zeros( img.size(), img.type() );
		for( int i=0; i< contours.size(); i++)
		{ 
			for( int j=0; j< contours[i].size(); j++)
			{
				circle( conturs, contours[i][j], 4, Scalar(255,255,255), 1 );
			}
		}
		imshow( "conturs", conturs );*/
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
