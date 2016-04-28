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

#define intrinsic_path	"/home/norbi/Asztal/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__intrinsic.xml"
#define	distortion_path	"/home/norbi/Asztal/object_recognition_c++/mobius_calibration_data/get_caloutput_20160413-2__distortion.xml"


#define LINE_MAX_DISTANCE 50	//In pixels
#define LINE_MIN_DISTANCE 10	//In pixels
#define	LINE_MAX_ANGLE	10	//In degrees
#define LINE_MIN_LENGHT	20	//In pixels

using namespace cv;

void readme();

/** @function main */
int main( int argc, char** argv )
{
	
	uint16_t save_sequence=0;


	enum { OK=0, 
		error_reading_images=1
		}	error=OK;

	if( argc != 2 )		{ readme(); return -1; }
	
  	Mat img_find = imread( argv[1]  );		//amit mekeresek a képben

	VideoCapture cap(1);
//	cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);

	

while(1){
	Mat img;
	bool bSuccess = cap.read(img);
	img = imread("/home/norbi/Asztal/object_recognition_c++/build/saved images/webcam_capture_2.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file, comment if use camera
	Mat img_orginal=img;
	
/*************************************** Undistortion code ***********************************************************************/

	#define crop_on
	//#define crop_off

	CvMat *intrinsic, *distortion;
	IplImage *inputImg, *outputImg;

	// Load parameters and image
	IplImage temp=img;	//convert Mat to IplImage	
	inputImg = &temp;
	outputImg = cvCreateImage( cvGetSize( inputImg ), inputImg->depth, 3 );
	intrinsic = (CvMat*)cvLoad( intrinsic_path );
	distortion = (CvMat*)cvLoad( distortion_path );

#ifdef	crop_on
//	printf("Crop On");
	cvUndistort2( inputImg, outputImg, intrinsic, distortion );
	img=outputImg;
#endif
#ifdef	crop_off
//	printf("Crop Off");
	double alpha=1;
	CvMat *cameraMatrix = cvCreateMat( 3, 3, CV_32FC1 );
	    IplImage *mapx = cvCreateImage( cvGetSize( inputImg ), IPL_DEPTH_32F, 1 );
	    IplImage *mapy = cvCreateImage( cvGetSize( inputImg ), IPL_DEPTH_32F, 1 );

	    cvGetOptimalNewCameraMatrix(
		intrinsic,
		distortion,
		cvGetSize( inputImg ),
		alpha,
		cameraMatrix,
		cvGetSize( inputImg )
	    );

	    cvInitUndistortRectifyMap(
		intrinsic,
		distortion,
		NULL,
		cameraMatrix,
		mapx,
		mapy
	    );

	    cvRemap( inputImg, outputImg, mapx, mapy );
	img=outputImg;
#endif
/********************************************************************************************************************************/
/*************************************** Convert image to gray, binary **********************************************************/

	Mat image_temp,image_temp_gray;
	image_temp=img;
	cvtColor(image_temp,image_temp_gray,CV_RGB2GRAY);
	Mat img_binary = image_temp_gray > 64;			//Itt valtozik binarissa a kep
	img_binary = ~img_binary;				//negállom a képet

	image_temp=img_find;
	cvtColor(image_temp,image_temp_gray,CV_RGB2GRAY);
	Mat img_find_binary = image_temp_gray > 64;			//Itt valtozik binarissa a kep
	img_find_binary = ~img_find_binary;				//negállom a képet
	
/********************************************************************************************************************************/
/*************************************** Hough Lines ****************************************************************************/
	Mat dst,cdst;
	Canny(img, dst, 50, 200, 3);		//Detect the edges of the image by using a Canny detector
	imshow( "dst", dst );	
	cvtColor(dst, cdst, CV_GRAY2BGR);	//Transform image into graysacele
//	imshow( "cdst", cdst );	

#define HOUGH_P
#ifdef HOUGH
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );
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
		line( cdst, pt1, pt2, Scalar(0,0,255), 2, CV_AA);
		}
#endif
#ifdef HOUGH_P				//HOUGH_P ben kartezian koordinatarendszer van, HOUGH ban pedig polár utannanézni lehet ezért nem mukoditt ez
	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI/180, 40, 20, 10 );
	for( size_t i = 0; i < lines.size(); i++ )
		{
		Vec4i l = lines[i];
		if(l[0]<500 && l[2]<500){		
			//line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, CV_AA);
			char str[5];
			sprintf(str,"%d", i);
			putText(cdst,str, Point((l[0]+l[2])/2, (l[1]+l[3])/2), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);		
			//printf("x:%d y:%d     x:%d y:%d\n", l[0], l[1],l[2],l[3]);
			}
		}
#endif
/*********************************************************************************************************************************/
/************************************** Convex Hull ******************************************************************************/




/*********************************************************************************************************************************/
/************************************** Match Template ***************************************************************************
	Mat matchTemplate_output_binary,img_find_match_template_binary;
	Mat matchTemplate_output,img_orginal_match_template,img_find_match_template;
	
	int result_cols =  img_orginal.cols - img_find.cols + 1;
	int result_rows = img_orginal.rows - img_find.rows + 1;
	cvtColor(img_orginal,img_orginal_match_template,CV_32FC1);
	cvtColor(img_find,img_find_match_template,CV_32FC1);
	matchTemplate_output.create( result_rows, result_cols, CV_32FC1 );

	matchTemplate(img_binary,img_find_binary,matchTemplate_output_binary,3);
	matchTemplate(img_orginal_match_template,img_find_match_template,matchTemplate_output,3);
/********************************************************************************************************************************/
/************************************** Convex hull and defect algorithms *******************************************************/
Mat src_gray,threshold_output;
	int thresh = 85;
	vector<vector<Point> > contours;

   	cvtColor( img, src_gray, CV_BGR2GRAY );
   	blur( src_gray, src_gray, Size(3,3) );
	threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );	/// Detect edges using Threshold
	imshow( "threshold_output", threshold_output );	
	vector<Vec4i> hierarchy;
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	vector<vector<Point> >hull( contours.size() );
	vector<vector<int> >hull_int( contours.size() );
	std::vector< std::vector<Vec4i> > defects( contours.size() );
	for( int i = 0; i < contours.size(); i++ )	{  
		convexHull( Mat(contours[i]), hull[i], false );
		convexHull( Mat(contours[i]), hull_int[i], false );
		if (contours[i].size() >3 )
		        {
			convexityDefects(contours[i], hull_int[i], defects[i]);
	        }
	}

	Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
	Mat drawing1 = Mat::zeros( threshold_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ ){
		drawContours( drawing, contours, i, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull, i, Scalar(0,255,0), 1, 8, vector<Vec4i>(), 0, Point() );
	}
/************************************** Detect algorithm ************************************************************************/
	for( int i = 0; i< contours.size(); i++ ){								//Rajzolás
		size_t count = contours[i].size();
		if( count <50 )	{printf("*\n"); continue;}
		vector<Vec4i>::iterator d=defects[i].begin();
		int j=0;
		int parallel=0,perpendicular1=0,perpendicular2=0,cross_lenght[8];
		Point point_candidate[4];
		while( d!=defects[i].end() ) {
			Vec4i& v=(*d);
			int startidx=v[0]; Point ptStart( contours[i][startidx] );
			int endidx=v[1]; Point ptEnd( contours[i][endidx] );
			int faridx=v[2]; Point ptFar( contours[i][faridx] );
			float depth = v[3] / 256;
			d++;

			line( drawing1, ptStart, ptEnd, Scalar(0,255,0), 1 );	//zöld
			line( drawing1, ptStart, ptFar, Scalar(255,255,0), 1 );	//sárga
			line( drawing1, ptEnd, ptFar, Scalar(0,255,255), 1 );	//kék
			circle( drawing1, ptFar, 4, Scalar(0,255,0), 1 );
			circle( drawing1, ptStart, 4, Scalar(255,0,0), 1 );
			circle( drawing1, ptEnd, 4, Scalar(0,0,255), 1 );
			char str[5];
			sprintf(str,"%d", j++);
			putText(drawing1,str, Point(((ptStart.x+ptFar.x)/2),((ptStart.y+ptFar.y)/2)), FONT_HERSHEY_PLAIN, 1.0,CV_RGB(255,255,0), 1.0);
			putText(drawing1,str, Point(((ptEnd.x+ptFar.x)/2),((ptEnd.y+ptFar.y)/2)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 1.0);
			putText(drawing1,str, Point(ptFar.x+5,ptFar.y+5), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 1.0);
			
			vector<Vec4i>::iterator dd=defects[i].begin();			
			int lenght=norm(ptFar-ptStart);
			float Angle1, Angle2;
			Angle1 = atan2(ptFar.y - ptStart.y, ptFar.x - ptStart.x) * 180.0 / CV_PI;
			Angle2 = atan2(ptFar.y - ptEnd.y, ptFar.x - ptEnd.x) * 180.0 / CV_PI;
			if(lenght<LINE_MIN_LENGHT)	{printf("SHORT, line_lenght: %d\n", lenght); continue;}
			if((abs(abs(Angle1)-abs(Angle2))-90)<LINE_MAX_ANGLE) {
				point_candidate[perpendicular1]=ptFar;
				cross_lenght[2*perpendicular1]=norm(ptFar-ptStart);
				cross_lenght[2*perpendicular1+1]=norm(ptEnd-ptFar);
				perpendicular1++;
				if(perpendicular1>4)	{printf("TOO MUCH perpendicular>4");	continue;}
			}
			
			printf("Angle1: %.2f	Angle2: %.2f	", Angle1,Angle2);
			printf("Angle12: %.2f	Lenght: %d	",(float)(abs(abs(Angle1)-abs(Angle2))-90),lenght);
			printf("parallel lines: %d	Perpendicular lines: %d\n",parallel,perpendicular1);
		}	
		printf("------- i: %d Parallel lines: %d		Perpendicular lines: %d ----------------------\n", i,parallel,perpendicular1);
		printf("\ncros.x:\n");
		for(int i=0;i<4;i++) printf("%d		cross_lenght: %d\n",point_candidate[i].x,cross_lenght[2*i]);
		printf("cros.y:\n");
		for(int i=0;i<4;i++) printf("%d		cross_lenght: %d\n",point_candidate[i].y,cross_lenght[2*i+1]);

		if(perpendicular1 == 4){
			float Angle[8];
			Angle[0] = atan2(point_candidate[0].y - point_candidate[1].y, point_candidate[0].x - point_candidate[1].x) * 180.0 / CV_PI;
			Angle[1] = atan2(point_candidate[1].y - point_candidate[2].y, point_candidate[1].x - point_candidate[2].x) * 180.0 / CV_PI;

			Angle[2] = atan2(point_candidate[1].y - point_candidate[2].y, point_candidate[1].x - point_candidate[2].x) * 180.0 / CV_PI;
			Angle[3] = atan2(point_candidate[2].y - point_candidate[3].y, point_candidate[2].x - point_candidate[3].x) * 180.0 / CV_PI;
				
			Angle[4] = atan2(point_candidate[2].y - point_candidate[3].y, point_candidate[2].x - point_candidate[3].x) * 180.0 / CV_PI;
			Angle[5] = atan2(point_candidate[3].y - point_candidate[0].y, point_candidate[3].x - point_candidate[0].x) * 180.0 / CV_PI;

			Angle[6] = atan2(point_candidate[3].y - point_candidate[0].y, point_candidate[3].x - point_candidate[0].x) * 180.0 / CV_PI;
			Angle[7] = atan2(point_candidate[0].y - point_candidate[1].y, point_candidate[0].x - point_candidate[1].x) * 180.0 / CV_PI;
				
			for(int i=0;i<4;i++)	if((abs(abs(Angle[2*i])-abs(Angle[2*i+1]))-90)<LINE_MAX_ANGLE) {perpendicular2++;}
				
				
			for(int i=0;i<4;i++)	printf("Angle1: %.2f	Angel2: %.2f	Anlge12: %.2f	perpendicular2: %d\n", Angle[2*i],Angle[2*i+1],(float)(abs(abs(Angle[2*i])-abs(Angle[2*i+1]))),perpendicular2);
				
			if(perpendicular2 == 4){
				Point cross;				
				cross.x=(point_candidate[0].x+point_candidate[1].x+point_candidate[2].x+point_candidate[3].x)/4;
				cross.y=(point_candidate[0].y+point_candidate[1].y+point_candidate[2].y+point_candidate[3].y)/4;
				circle( drawing1, cross, 2, Scalar(0,0,255), 5 );
				circle( img, cross, 2, Scalar(255,0,0), 5 );
			}
		}
	}
	imshow( "Hull demo", drawing );
	imshow( "Hull demo1", drawing1 );
/**************************************************************************	******************************************************/
/************************************** Detect algorithm ************************************************************************/
	//Ez a rész összehasonlítja az egyes vonalak távolságát egymással. Ha meszebb vannak mint LINE_DISTANCE pixelekben nem foglalkozom tovább velük	
	float Angle1, Angle2;
	for(size_t i = 0; i < lines.size(); i++){
		Vec4i l = lines[i];
		Angle1 = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
		for(size_t j = i; j < lines.size(); j++){
			if(i!=j){			
				Vec4i ll = lines[j];			
				Angle2 = atan2(ll[3] - ll[1], ll[2] - ll[0]) * 180.0 / CV_PI;
				if(abs(Angle1-Angle2)<LINE_MAX_ANGLE){
					int line_distance=pow(pow(abs((ll[0] + ll[2])/2-(l[0] + l[2])/2),2) + pow(abs((ll[1] + ll[3])/2-(l[1] + l[3])/2),2),0.5);
					if(LINE_MAX_DISTANCE>line_distance && line_distance>LINE_MIN_DISTANCE){
					/*	printf(" i:%d	x1:%d	y1:%d	x2:%d	y2:%d	(x1+x2)/2:%d	(y1+y2)/2:%d	|	j:%d	x1:%d	y1:%d	x2:%d	y2:%d	(x1+x2)/2:%d	(y1+y2)/2:%d	|	a:%d	b:%d	r:%d	Angle_i:%f	Angle_j:%f",	
						i,l[0],l[1],l[2],l[3],(l[0]+l[2])/2,(l[1] + l[3])/2,j,ll[0],ll[1],ll[2],ll[3],(ll[0] + ll[2])/2,(ll[1] + ll[3])/2,abs((ll[0] + ll[2])/2-(l[0]+l[2])/2),
						abs((ll[1] + ll[3])/2-(l[1] + l[3])/2),(int)pow(pow(abs((ll[0] + ll[2])/2-(l[0]+l[2])/2),2) + pow(abs((ll[1] + ll[3])/2-(l[1] + l[3])/2),2),0.5),Angle1,Angle2);
						printf("\n");*/
						//Kiírom a Hough algoritmus által detektált vonal i számát, egyik pontjának az x1,y1 majd a másik pontjának x2,y2 koordinátáját, közepének az x,y koordinátáit, 
						//majd ugyanezt a j számú vonalra amivel hasonlítom össze.
						//Végül a két vonál közepének a(x - tengelyen) és b(y - tengelyen) vett távolságát majd a relatív távolságukat r=gyōk(a²+b²)

				
						line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);	
					}
				}
			}
		}
		//printf("------------------------------------\n");
	}




/********************************************************************************************************************************/
	char c=cvWaitKey(10);
	
//	imshow( "Orignal_image", img_orginal );
	imshow( "Undistorted_image", img );
	imshow( "cdst", cdst );
//	imshow( "img_find", img_find );
//	imshow( "matchTemplate", matchTemplate_output );
//	imshow( "matchTemplate_binary", matchTemplate_output_binary );
	
	if(c== 'a')	{char filename[25]; snprintf(filename,24,"webcam_capture_%d.jpg",save_sequence); printf("image saved"); imwrite(filename, img_orginal); save_sequence++;}
	//press 'a' button for save picture
}

/********************************************************************************SURF********************************************/
	while(1){Mat imgOriginal;bool bSuccess = cap.read(imgOriginal);
//	Mat imgOriginal = img;


	cv::cvtColor(imgOriginal, imgOriginal, cv::COLOR_BGR2GRAY);

	if (!bSuccess) //if not success, break loop
        {
             printf("read error");
             break;
        }
//	CvMat* prevgray = 0, *image = 0, *gray =0;
//	CvCapture* capture = cvCreateCameraCapture(1);
//	IplImage* frame = cvQueryFrame(capture);
//	image = cvCreateMat(frame->height, frame->width, CV_8UC1);
//	cvShowImage( "Image", frame );  	
//	imshow( "test_camera", imgOriginal );

	if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
       {
            std::cout << "esc key is pressed by user" << std::endl;
            break; 
       }

  	Mat img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );		//amit mekeresek a képben
  	Mat img_scene = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );		//amiben keresem az objektumot
	Mat imgHSV;

 //	cvtColor(imgOriginal, img_scene, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
	img_scene=imgOriginal;
  	if( !img_object.data || !img_scene.data ){
		error=error_reading_images;
		printf("error: %d\nSee enumerated values \n", error_reading_images);
		return error_reading_images;
		}

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 500;

  SurfFeatureDetector detector( minHessian );
  
  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;
printf(" %d\n", descriptors_object.rows);
	int cnt_matches=0;
	for( int i = 0; i < descriptors_object.rows; i++ )
	  { if( matches[i].distance < 1.2*min_dist )
	     { good_matches.push_back( matches[i]); cnt_matches++; }
	  }
/*	good_matches.push_back( matches[0]);
	good_matches.push_back( matches[1]);
	good_matches.push_back( matches[2]);
	good_matches.push_back( matches[3]);*/

  Mat img_matches, img_own;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	if(cnt_matches<8) {
		imshow( "Good Matches & Object detection", img_matches );
  		//imshow( "test", img_scene );
		
	}
else{
  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }
std::cout << "debug 3\n" << std::endl;

	Mat H;
if(cnt_matches>4) H = findHomography( obj, scene, CV_RANSAC );
	std::cout << "debug 1\n" << std::endl;
  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
 std::cout << "debug 2\n" << std::endl;
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//  line( img_scene, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 0, 230), 4 );
//  line( img_scene, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//  line( img_scene, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 50, 255, 0), 4 );
//  line( img_scene, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 10 );
  
  circle( img_matches,								//bal felső és a jobb alsó sarok közé középre egy pont rajzolása
         (scene_corners[0] + scene_corners[2])*.5 + Point2f( img_object.cols, 0), //pont helye
         5,											//nagyság
         Scalar( 255, 0, 0 ),						//pont szine
         -1,										//-1 egész kör kitöltése, amugy a körív vastagsága
         CV_AA );									//körvonál típusa

  circle( img_matches,								//jobb felső és a bal alsó sarok közé középre egy pont rajzolása
         (scene_corners[1] + scene_corners[3])*.5 + Point2f( img_object.cols, 0),
         5,
         Scalar( 255, 255, 0 ),
         -1,
         2 );

  circle( img_scene,								//a négy sarok határolta középre egy pont rajzolása
         (scene_corners[0] + scene_corners[1] + scene_corners[2] + scene_corners[3])*.25,	//kör helye
         10,										//nagyság
         Scalar( 255, 255, 255 ),					//kör szine
         -1,										//-1 egész kör kitöltése, amugy a körív vastagsága
         CV_AA );										//körvonál típusa

  printf("az objektum koordinátáááái: \n");
  printf("x : %f \n", ((scene_corners[0] + scene_corners[1] + scene_corners[2] + scene_corners[3])*.25).x );
  printf("y : %f \n", ((scene_corners[0] + scene_corners[1] + scene_corners[2] + scene_corners[3])*.25).y );

  //-- Show detected matches
 // imshow( "Good Matches & Object detection", img_matches );
  //imshow( "test", img_scene );
}
}
  waitKey(0);
  return 0;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }
