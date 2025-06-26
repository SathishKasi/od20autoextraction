// dimNoteDimRecog.cpp : Defines the exported functions for the DLL application.

#include <windows.h>
#include <shobjidl.h> 
#include <vector>
#include <algorithm>
#include "dimNoteDimRecog.h"
#include "stdafx.h"
#include <stdexcept>


#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

using namespace cv;
using namespace std;
///////////////////////////////////////////////////////////////////////////////////////////////////
namespace dimNoteDimRecog
{
	opVector output;
	int iterIndex;
	int kNN;
	DimSet dummyDimSet;
	Point2f dummyCenter;
	double trialImageWidth;
	double trialImageHeight;
	cv::Mat inputImage;
	cv::Mat trialImage;
	Mat dst;
	bool selectObject = false;
	Rect selection;
	Point origin;
	Mat image;
	bool insert = false;

	Rect PatchRect;
	Mat PatchImg;
	const int RESIZED_IMAGE_WIDTH = 20;
	const int RESIZED_IMAGE_HEIGHT = 30;
	///////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();//creating a kNearest object
	///////////////////////////////////////////////////////////////////////////////////////////////////
	class newDimSet
	{
	public:
		Mat newDimSetImg;
		double x;
		double y;
		double width;
		double height;
	};
	std::vector <newDimSet> setOfnewdimGDTNoteSet;
	///////////////////////////////////////////////////////////////////////////////////////////////////
	class ContourWithData {
	public:
		// member variables ///////////////////////////////////////////////////////////////////////////
		std::vector<cv::Point> ptContour;           // contour
		cv::Rect boundingRect;                      // bounding rect for contour
		float fltArea;                              // area of contour

		bool checkIfContourIsValid() {                              
			if (fltArea < MIN_CONTOUR_AREA) return false;           
			return true;                                           
		}
		///////////////////////////////////////////////////////////////////////////////////////////////
		static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      
			return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                  
		}
	};
	///////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat maximizeContrast(cv::Mat &imageDimSetGSBW) {
		cv::Mat imgTopHat;
		cv::Mat imgBlackHat;
		cv::Mat imageDimSetGSBWPlusTopHat;
		cv::Mat imageDimSetGSBWPlusTopHatMinusBlackHat;

		cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(2, 2));

		cv::morphologyEx(imageDimSetGSBW, imgTopHat, CV_MOP_TOPHAT, structuringElement);
		cv::morphologyEx(imageDimSetGSBW, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

		imageDimSetGSBWPlusTopHat = imageDimSetGSBW + imgTopHat;
		imageDimSetGSBWPlusTopHatMinusBlackHat = imageDimSetGSBWPlusTopHat - imgBlackHat;

		return(imageDimSetGSBWPlusTopHatMinusBlackHat);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat lineCleanse(Mat &lineCleanseImg)// Eliminate horizontal and vertical lines
	{
		Mat gray = lineCleanseImg;

		if (lineCleanseImg.channels() == 3)
		{
			cvtColor(gray, gray, CV_BGR2GRAY);
		}
		else
		{
			gray = lineCleanseImg;
		}
		if (gray.empty())
		{
			exit(-1);
		}
		Mat bw;

		adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
		// Create the images that will use to extract the horizontal and vertical lines
		Mat horizontal = bw.clone();
		Mat vertical = bw.clone();
		Mat horVer;
		// Specify size on horizontal axis
		int horizontalsize = horizontal.cols / 100;
		// Create structure element for extracting horizontal lines through morphology operations
		Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
		// Apply morphology operations
		erode(horizontal, horizontal, horizontalStructure, Point(-1, -1), 5);
		dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1), 10);
		// Specify size on vertical axis
		int verticalsize = vertical.rows / 100;
		Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
		erode(vertical, vertical, verticalStructure, Point(-1, -1), 5);
		dilate(vertical, vertical, verticalStructure, Point(-1, -1), 10);

		horVer = horizontal + vertical;//The determined lines are on an image 
		subtract(bw, horVer, horVer);// the resultant line removed image

		GaussianBlur(horVer, horVer, Size(9, 9), 2, 2);
		vector<Vec3f> circles;
		HoughCircles(horVer, circles, CV_HOUGH_GRADIENT, 1, horVer.rows / 4, 200, 100, horVer.rows / 3, 0);
		/// Draw the detected circles on the input image
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(horVer, center, radius, SCALAR_BLACK, 10, 8, 0);
		}

		bitwise_not(horVer, horVer);

		horVer = maximizeContrast(horVer);// maximise contrast to highlight the chracteristics
		cvtColor(horVer, horVer, CV_GRAY2BGR);

		return horVer;
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	double compute_skew(cv::Mat toRotImg)
	{
		toRotImg = maximizeContrast(toRotImg);

		cv::Mat src = toRotImg;
		cvtColor(src, src, CV_BGR2GRAY);//Convert to greyscale image
		cv::Size size = src.size();

		cv::bitwise_not(src, src);

		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(src, lines, 1, CV_PI / 180, 50, size.width / 2.f, 20);//Determine the lines by probabilistic hough transform

		cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));
		double angle = 0.;
		for (unsigned i = 0; i < lines.size(); ++i)
		{
			cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
				cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0));
			angle += atan2((double)lines[i][3] - lines[i][1],
				(double)lines[i][2] - lines[i][0]);
		}
		angle /= lines.size(); // mean angle, in radians.
		angle = angle * 180 / CV_PI;// mean angle, in degrees.
		return(angle);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	double check_90deg(cv::Mat toRotImg)
	{
		Mat gray = toRotImg;

		if (toRotImg.channels() == 3)
		{
			cvtColor(gray, gray, CV_BGR2GRAY);//Convert to greyscale if the image is color
		}
		else
		{
			gray = toRotImg;
		}
		if (gray.empty())
		{
			exit(-1);
		}

		cv::Mat src = gray;

		cv::Size size = src.size();

		cv::bitwise_not(src, src);

		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(src, lines, 1, CV_PI / 180, 50, size.width / 2.f, 20);//Determine the lines by probabilistic hough transform

		cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));
		double angle = 0.;
		for (unsigned i = 0; i < lines.size(); ++i)
		{
			cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
			cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0));

			if ((lines[i][2] - lines[i][0]) < (src.size().height / 10))//Check if the lines are nearly vertical by checking how close the starting X coordinates of two lines are.
			{
				angle += CV_PI / 2;//If so assign pi/2 (radians value of 90 degrees) to the angle 
			}
			else
			{
				angle += CV_PI / 4;
			}
		}
		angle /= lines.size(); // mean angle, in radians.
		angle = angle * 180 / CV_PI;// mean angle, in degrees.
		return(angle);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat deskew(cv::Mat src)
	{
		src = maximizeContrast(src);
		double angle = 0;
		cv::Mat rotated;
		if (src.channels() == 3)
		{
			cvtColor(src, src, CV_BGR2GRAY);//Convert to greyscale if the image is color
		}
		else
		{
			src = src;
		}
		if (src.empty())
		{
			exit(-1);
		}
		cv::Size size = src.size();
		cv::bitwise_not(src, src);

		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(src, lines, 1, CV_PI / 180, 50, size.width / 2.f, 20);//Determine the lines by probabilistic hough transform

		cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));

		for (unsigned i = 0; i < lines.size(); ++i)
		{
			cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
				cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0));
			angle += atan2((double)lines[i][3] - lines[i][1],
				(double)lines[i][2] - lines[i][0]);
		}
		angle /= lines.size(); // mean angle, in radians.
		angle = angle * 180 / CV_PI;// mean angle, in degrees.		

/*		std::vector<cv::Point> points;
		cv::Mat_<uchar>::iterator it = toRotImg.begin<uchar>();
		cv::Mat_<uchar>::iterator end = toRotImg.end<uchar>();
		for (; it != end; ++it)
			if (*it)
				points.push_back(it.pos()); //determine points to be crop out to a selected region image

		cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));*/

		cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(src.rows/2, src.cols), angle, 1);//determine the rotation matrix 
		cv::warpAffine(src, rotated, rot_mat, Size((3 * src.rows), (3 * src.cols)), cv::INTER_CUBIC, cv::BORDER_CONSTANT, SCALAR_BLACK);// rotate the image
		return rotated;//return rotated image
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	static void onMouse(int event, int x, int y, int, void*)
	{
		newDimSet newDimNoteSet;
		newDimNoteSet.x = 0;
		newDimNoteSet.y = 0;
		newDimNoteSet.width = 0;
		newDimNoteSet.newDimSetImg = Mat(image, Rect(0, 0, 0, 0));//declaring a dummy to prevent garbage values

		if (selectObject & insert)
		{
			selection.x = MIN(x, origin.x);
			selection.y = MIN(y, origin.y);
			selection.width = std::abs(x - origin.x);
			selection.height = std::abs(y - origin.y);
			selection &= Rect(0, 0, image.cols, image.rows);
		}

		switch (event)
		{
		case CV_EVENT_LBUTTONDOWN:
			origin = Point(x, y);
			selection = Rect(x, y, 0, 0);
			selectObject = true;
			break;
		case CV_EVENT_LBUTTONUP:
			if (selectObject && insert)
			{
				if (selection.width > 5 && selection.height > 5)
				{
					newDimNoteSet.x = selection.x;
					newDimNoteSet.y = selection.y;
					newDimNoteSet.width = selection.width;
					newDimNoteSet.height = selection.height;//assigning the selected region to a new dimension or note set. 
					PatchRect = selection;
					image(PatchRect).copyTo(PatchImg);//crop out selected region from image
					imshow("Selected Img", PatchImg);
					waitKey(0);
					newDimNoteSet.newDimSetImg = PatchImg;
					setOfnewdimGDTNoteSet.push_back(newDimNoteSet);//storing the new dimension or note set. 
				}
				else
					selection = Rect(0, 0, 0, 0);
			}
			selectObject = false;
			insert = false;
			break;
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	Character::Character(std::vector<cv::Point> _contour) {
		contour = _contour;

		boundingRect = cv::boundingRect(contour);

		intCenterX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
		intCenterY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

		dblDiagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));

		dblAspectRatio = (float)boundingRect.width / (float)boundingRect.height;
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat extractValue(cv::Mat &imgOriginal) {
		cv::Mat imgHSV;
		std::vector<cv::Mat> vectorOfHSVImages;
		cv::Mat imgValue;

		cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);

		cv::split(imgHSV, vectorOfHSVImages);

		imgValue = vectorOfHSVImages[2];

		return(imgValue);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	void preprocess(cv::Mat &imgOriginal, cv::Mat &imageDimSetGSBW, cv::Mat &imageDimSetThreshold) {
		imageDimSetGSBW = extractValue(imgOriginal);                           // extract value channel only from original image to get imageDimSetGSBW

		cv::Mat imgMaxContrastGrayscale = maximizeContrast(imageDimSetGSBW);       // maximize contrast with top hat and black hat

		cv::Mat imgBlurred;

		cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);          // gaussian blur

		cv::adaptiveThreshold(imgBlurred, imageDimSetThreshold, 255.0, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	//                                        Detect Dimensions or Notes:                            //
	///////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<DimSet> detectDimSet(cv::Mat &inputImage) {
		std::vector<DimSet> vectorOfdimGDTNoteSet;			// this will be the return value

		cv::Mat imageDimSetGSBWScene;
		cv::Mat imageDimSetThresholdScene;

		preprocess(inputImage, imageDimSetGSBWScene, imageDimSetThresholdScene);        // preprocess to get grayscale and threshold images
																					
		std::vector<Character> vectorOfCharactersInScene = findCharactersFromInput(imageDimSetThresholdScene);
		std::vector<std::vector<cv::Point> > contours;

		for (auto &Character : vectorOfCharactersInScene) {
			contours.push_back(Character.contour);
		}
		// given a vector of all possible chars, find groups of matching chars
		std::vector<std::vector<Character> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfCharactersInScene);

		for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene)
		{
			std::vector<std::vector<cv::Point> > contours;
			for (auto &matchingChar : vectorOfMatchingChars)
			{
				contours.push_back(matchingChar.contour);
			}
		}
		for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {                     // for each group of matching chars
			DimSet posDimSet = extractDimSet(inputImage, vectorOfMatchingChars);        // attempt to extract dimGDTNoteSet

			if (posDimSet.imageDimSet.empty() == false) {                                              // if dimGDTNoteSet was found
				vectorOfdimGDTNoteSet.push_back(posDimSet);                                        // add to vector of possible dimGDTNoteSet
			}
		}
		for (unsigned int i = 0; i < vectorOfdimGDTNoteSet.size(); i++) {
			cv::Point2f p2fRectPoints[4];
			vectorOfdimGDTNoteSet[i].dimSetLocations.points(p2fRectPoints);
		}
		return vectorOfdimGDTNoteSet;
	}
///////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<Character> findCharactersFromInput(cv::Mat &imageDimSetThreshold) {
		std::vector<Character> vectorOfCharacters;            // this will be the return value
		int intCountOfCharacters = 0;

		cv::Mat imageDimSetThresholdCopy = imageDimSetThreshold;

		std::vector<std::vector<cv::Point> > contours;
		vector<Vec4i> hierarchy;

		cv::findContours(imageDimSetThresholdCopy, contours, hierarchy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE);        // find all contours

		for (unsigned int i = 0; i < contours.size(); i++)
		{                // for each contour
			Character charac(contours[i]);

			if (checkIfCharacter(charac))
			{                // if contour is a possible char, note this does not compare to other chars (yet) . . .
				if (hierarchy[i][3] == -1)// only consider contours that do not have a parent
				{
					intCountOfCharacters++;                          // increment count of possible chars
					vectorOfCharacters.push_back(charac);      // and add to vector of possible chars
				}
			}
		}
		//std::cout << std::endl << "contours.size() = " << contours.size() << std::endl;                        
		//std::cout << "CountOfValidCharacters = " << intCountOfCharacters << std::endl;     																			 
		return(vectorOfCharacters);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	DimSet extractDimSet(cv::Mat &imgOriginal, std::vector<Character> &vectorOfMatchingChars) {
		DimSet DimSet;            // this will be the return value
								  // sort chars from left to right based on x position
		std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), Character::sortCharsLeftToRight);
		dummyCenter.x = 1;
		dummyCenter.y = 1;
		float 	dummyAngle = 0;

		dummyDimSet.dimSetLocations = cv::RotatedRect(dummyCenter, cv::Size2f(1, 1), dummyAngle);
		dummyDimSet.dimNoteStrings = "dummy";

		// calculate the center point of the dimGDTNoteSet
		double dbldimGDTNoteSetCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
		double dbldimGDTNoteSetCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;
		cv::Point2d p2ddimGDTNoteSetCenter(dbldimGDTNoteSetCenterX, dbldimGDTNoteSetCenterY);
		// calculate dimGDTNoteSet width and height
		int intdimGDTNoteSetWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * dimGDTNoteSet_WIDTH_PADDING_FACTOR);

		double intTotalOfCharHeights = 0;

		for (int i = 0; i < vectorOfMatchingChars.size(); i++)
		{
			intTotalOfCharHeights = intTotalOfCharHeights + vectorOfMatchingChars[i].boundingRect.height;
		}

		double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();

		int intdimGDTNoteSetHeight = (int)(dblAverageCharHeight * dimGDTNoteSet_HEIGHT_PADDING_FACTOR);
		// calculate correction angle of dimGDTNoteSet region
		double dblOpposite = vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY - vectorOfMatchingChars[0].intCenterY;
		double dblHypotenuse = distanceBetweenChars(vectorOfMatchingChars[0], vectorOfMatchingChars[vectorOfMatchingChars.size() - 1]);
		double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
		double dblCorrectionAngleInDeg = 0;
		if (iterIndex == 4)
		{
			dblCorrectionAngleInDeg = (dblCorrectionAngleInRad * (180.0 / CV_PI)) + 180;
		}
		if (iterIndex < 4)
		{
			dblCorrectionAngleInDeg = (dblCorrectionAngleInRad * (180.0 / CV_PI));
		}
		// assign rotated rect member variable of possible dimGDTNoteSet
		int intdimGDTNoteSetArea = intdimGDTNoteSetHeight*intdimGDTNoteSetWidth;

		if (intdimGDTNoteSetArea < ((trialImageWidth*trialImageHeight) / 16))
		{
			if (intdimGDTNoteSetHeight < trialImageHeight / 20)
			{
				DimSet.dimSetLocations = cv::RotatedRect(p2ddimGDTNoteSetCenter, cv::Size2f(((float)intdimGDTNoteSetWidth), (float)intdimGDTNoteSetHeight), (float)dblCorrectionAngleInDeg);
			}
			else
			{
				DimSet.dimSetLocations = dummyDimSet.dimSetLocations;
			}

		}
		else
		{
			DimSet.dimSetLocations = dummyDimSet.dimSetLocations;
		}

		cv::Mat rotationMatrix;             // final steps are to perform the actual rotation
		cv::Mat imgRotated;
		cv::Mat imgCropped;

		rotationMatrix = cv::getRotationMatrix2D(p2ddimGDTNoteSetCenter, dblCorrectionAngleInDeg, 1.0);         // get the rotation matrix for our calculated correction angle

		cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());            // rotate the entire image
																										// crop out the actual dimGDTNoteSet portion of the rotated image
		cv::getRectSubPix(imgRotated, DimSet.dimSetLocations.size, DimSet.dimSetLocations.center, imgCropped);

		DimSet.imageDimSet = imgCropped;            // copy the cropped dimGDTNoteSet image into the applicable member variable of the possible dimGDTNoteSet

		return(DimSet);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	//                                        Detect Characs                                         //
	///////////////////////////////////////////////////////////////////////////////////////////////////
	bool loadKNNDataAndTrainKNN(void) {		// read in training classifications ///////////////////////////////////////////////////
		cv::Mat matClassificationInts;              // we will read the classification numbers into this variable as though it is a vector

		cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

		if (fsClassifications.isOpened() == false) {                                                        // if the file was not opened successfully
			std::cout << "error, unable to open training classifications file, exiting program\n\n";        // show error message
			return(false);                                                                                  // and exit program
		}

		fsClassifications["classifications"] >> matClassificationInts;          // read classifications section into Mat classifications variable
		fsClassifications.release();                                            // close the classifications file
																				// read in training images ////////////////////////////////////////////////////////////
		cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

		cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);              // open the training images file

		if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
			std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
			return(false);                                                                          // and exit program
		}
		fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
		fsTrainingImages.release();                                                 // close the traning images file

		kNearest->setDefaultK(kNN);// finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat) even though in reality they are multiple images / numbers
		kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);
		return true;
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<DimSet> detectChars(std::vector<DimSet> &vectorOfdimGDTNoteSet) {
		// this is only for showing steps
		std::vector<std::vector<cv::Point> > contours;

		if (vectorOfdimGDTNoteSet.empty()) {               // if vector of possible dimGDTNoteSet is empty
			return(vectorOfdimGDTNoteSet);                 // return
		}
		// at this point we can be sure vector of possible dimGDTNoteSet has at least one dimGDTNoteSet

		for (auto &DimSet : vectorOfdimGDTNoteSet) {            // for each possible dimGDTNoteSet, this is a big for loop that takes up most of the function

			preprocess(DimSet.imageDimSet, DimSet.imageDimSetGSBW, DimSet.imageDimSetThreshold);        // preprocess to get grayscale and threshold images
																													// upscale size by 60% for better viewing and character recognition
			cv::resize(DimSet.imageDimSetThreshold, DimSet.imageDimSetThreshold, cv::Size(), 5, 5);

			// threshold again to eliminate any gray areas
			cv::threshold(DimSet.imageDimSetThreshold, DimSet.imageDimSetThreshold, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);

			// find all possible chars in the dimGDTNoteSet,
			// this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
			std::vector<Character> vectorOfCharactersIndimGDTNoteSet = findChars(DimSet.imageDimSetGSBW, DimSet.imageDimSetThreshold);
			contours.clear();

			for (auto &Character : vectorOfCharactersIndimGDTNoteSet) {
				contours.push_back(Character.contour);
			}

			// given a vector of all possible chars, find groups of matching chars within the dimGDTNoteSet
			std::vector<std::vector<Character> > vectorOfVectorsOfMatchingCharsIndimGDTNoteSet = findVectorOfVectorsOfMatchingChars(vectorOfCharactersIndimGDTNoteSet);

			//	vectorOfCharactersIndimGDTNoteSet<<Character>>
			contours.clear();

			for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsIndimGDTNoteSet) {
				for (auto &matchingChar : vectorOfMatchingChars) {
					contours.push_back(matchingChar.contour);
				}
			}

			if (vectorOfVectorsOfMatchingCharsIndimGDTNoteSet.size() == 0) {                // if no groups of matching chars were found in the dimGDTNoteSet
				dummyCenter.x = 1;
				dummyCenter.y = 1;
				float 	dummyAngle = 0;

				DimSet.dimSetLocations = cv::RotatedRect(dummyCenter, cv::Size2f(1, 1), dummyAngle);
				continue;                               // go back to top of for loop
			}

			for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsIndimGDTNoteSet) {                                         // for each vector of matching chars in the current dimGDTNoteSet
				std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), Character::sortCharsLeftToRight);      // sort the chars left to right
				vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                     // and eliminate any overlapping chars
			}

			for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsIndimGDTNoteSet) {
				contours.clear();
				for (auto &matchingChar : vectorOfMatchingChars) {
					contours.push_back(matchingChar.contour);
				}
			}
			// within each possible dimGDTNoteSet, suppose the longest vector of potential matching chars is the actual vector of chars
			unsigned int intLenOfLongestVectorOfChars = 0;
			unsigned int intIndexOfLongestVectorOfChars = 0;
			// loop through all the vectors of matching chars, get the index of the one with the most chars
			for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsIndimGDTNoteSet.size(); i++) {
				if (vectorOfVectorsOfMatchingCharsIndimGDTNoteSet[i].size() > intLenOfLongestVectorOfChars) {
					intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsIndimGDTNoteSet[i].size();
					intIndexOfLongestVectorOfChars = i;
				}
			}
			// suppose that the longest vector of matching chars within the dimGDTNoteSet is the actual vector of chars
			std::vector<Character> longestVectorOfMatchingCharsIndimGDTNoteSet = vectorOfVectorsOfMatchingCharsIndimGDTNoteSet[intIndexOfLongestVectorOfChars];

			contours.clear();

			for (auto &matchingChar : longestVectorOfMatchingCharsIndimGDTNoteSet) {
				contours.push_back(matchingChar.contour);
			}
			// perform char recognition on the longest vector of matching chars in the dimGDTNoteSet
			std::vector<std::string>vectorStrings = recognizeCharsInDimSet(DimSet.imageDimSetThreshold, longestVectorOfMatchingCharsIndimGDTNoteSet);
			DimSet.dimNoteStrings = vectorStrings[0];
			DimSet.accuracyPercentage = vectorStrings[1];
			DimSet.confidencePercentage = vectorStrings[2];
		}   
		return(vectorOfdimGDTNoteSet);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<Character> findChars(cv::Mat &imageDimSetGSBW, cv::Mat &imageDimSetThreshold) {
		std::vector<Character> vectorOfCharacters;                            // this will be the return value

		cv::Mat imageDimSetThresholdCopy;

		std::vector<std::vector<cv::Point> > contours;

		imageDimSetThresholdCopy = imageDimSetThreshold.clone();				// make a copy of the thresh image, this in necessary b/c findContours modifies the image

		cv::findContours(imageDimSetThresholdCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);        // find all contours in dimGDTNoteSet

		for (auto &contour : contours) {                            // for each contour
			Character charac(contour);
			if (checkIfCharacter(charac)) {                // if contour is a possible char, note this does not compare to other chars (yet) . . .
				vectorOfCharacters.push_back(charac);      // add to vector of possible chars
			}
		}
		return(vectorOfCharacters);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	bool checkIfCharacter(Character &Character) {
		// this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
		// note that we are not (yet) comparing the char to other chars to look for a group
		if (Character.boundingRect.area() > MIN_PIXEL_AREA &&
			Character.boundingRect.width > MIN_PIXEL_WIDTH && Character.boundingRect.height > MIN_PIXEL_HEIGHT &&
			MIN_ASPECT_RATIO < Character.dblAspectRatio && Character.dblAspectRatio < MAX_ASPECT_RATIO) {
			return(true);
		}
		else {
			return(false);
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<Character> > findVectorOfVectorsOfMatchingChars(const std::vector<Character> &vectorOfCharacters) {
		// with this function, we start off with all the possible chars in one big vector
		// the purpose of this function is to re-arrange the one big vector of chars into a vector of vectors of matching chars,
		// note that chars that are not found to be in a group of matches do not need to be considered further
		std::vector<std::vector<Character> > vectorOfVectorsOfMatchingChars;             // this will be the return value

		for (auto &charac : vectorOfCharacters) {                  // for each possible char in the one big vector of chars
																			   // find all chars in the big vector that match the current char
			std::vector<Character> vectorOfMatchingChars = findVectorOfMatchingChars(charac, vectorOfCharacters);
			vectorOfMatchingChars.push_back(charac);          // also add the current char to current possible vector of matching chars
																		  // if current possible vector of matching chars is not long enough to constitute a possible dimGDTNoteSet
			if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) {
				continue;                       // jump back to the top of the for loop and try again with next char, note that it's not necessary
												// to save the vector in any way since it did not have enough chars to be a possible dimGDTNoteSet
			}
			// if we get here, the current vector passed test as a "group" or "cluster" of matching chars
			vectorOfVectorsOfMatchingChars.push_back(vectorOfMatchingChars);            // so add to our vector of vectors of matching chars
																						// remove the current vector of matching chars from the big vector so we don't use those same chars twice,
																						// make sure to make a new big vector for this since we don't want to change the original big vector
			std::vector<Character> vectorOfCharactersWithCurrentMatchesRemoved;

			for (auto &possChar : vectorOfCharacters) {
				if (std::find(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possChar) == vectorOfMatchingChars.end()) {
					vectorOfCharactersWithCurrentMatchesRemoved.push_back(possChar);
				}
			}
			// declare new vector of vectors of chars to get result from recursive call
			std::vector<std::vector<Character> > recursiveVectorOfVectorsOfMatchingChars;

			// recursive call
			recursiveVectorOfVectorsOfMatchingChars = findVectorOfVectorsOfMatchingChars(vectorOfCharactersWithCurrentMatchesRemoved);	// recursive call !!

			for (auto &recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) {      // for each vector of matching chars found by recursive call
				vectorOfVectorsOfMatchingChars.push_back(recursiveVectorOfMatchingChars);               // add to our original vector of vectors of matching chars
			}
			break;		// exit for loop
		}
		return(vectorOfVectorsOfMatchingChars);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<Character> findVectorOfMatchingChars(const Character &charac, const std::vector<Character> &vectorOfChars) {
		// the purpose of this function is, given a possible char and a big vector of possible chars,
		// find all chars in the big vector that are a match for the single possible char, and return those matching chars as a vector
		std::vector<Character> vectorOfMatchingChars;                // this will be the return value

		for (auto &possibleMatchingChar : vectorOfChars) {              // for each char in big vector
																		// if the char we attempting to find matches for is the exact same char as the char in the big vector we are currently checking
			if (possibleMatchingChar == charac) {
				// then we should not include it in the vector of matches b/c that would end up double including the current char
				continue;           // so do not add to vector of matches and jump back to top of for loop
			}
			// compute stuff to see if chars are a match
			double dblDistanceBetweenChars = distanceBetweenChars(charac, possibleMatchingChar);
			double dblAngleBetweenChars = angleBetweenChars(charac, possibleMatchingChar);
			double dblChangeInArea = (double)abs(possibleMatchingChar.boundingRect.area() - charac.boundingRect.area()) / (double)charac.boundingRect.area();
			double dblChangeInWidth = (double)abs(possibleMatchingChar.boundingRect.width - charac.boundingRect.width) / (double)charac.boundingRect.width;
			double dblChangeInHeight = (double)abs(possibleMatchingChar.boundingRect.height - charac.boundingRect.height) / (double)charac.boundingRect.height;
			// check if chars match
			if ((iterIndex == 2) || (iterIndex == 4))
			{
				if (dblDistanceBetweenChars < (charac.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) && abs(dblAngleBetweenChars) >15 &&
					abs(dblAngleBetweenChars) < 180 &&
					dblChangeInArea < MAX_CHANGE_IN_AREA &&
					dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&
					dblChangeInHeight < MAX_CHANGE_IN_HEIGHT)
				{
					vectorOfMatchingChars.push_back(possibleMatchingChar);      // if the chars are a match, add the current char to vector of matching chars
				}
			}
			else
			{
				if (dblDistanceBetweenChars < (charac.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) &&
					abs(dblAngleBetweenChars) < 15 &&
					dblChangeInArea < MAX_CHANGE_IN_AREA &&
					dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&
					dblChangeInHeight < MAX_CHANGE_IN_HEIGHT)
				{
					vectorOfMatchingChars.push_back(possibleMatchingChar);      // if the chars are a match, add the current char to vector of matching chars
				}
			}
		}
		return(vectorOfMatchingChars);          // return result
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// use Pythagorean theorem to calculate distance between two chars
	double distanceBetweenChars(const Character &firstChar, const Character &secondChar) {
		int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
		int intY = abs(firstChar.intCenterY - secondChar.intCenterY);
		return(sqrt(pow(intX, 2) + pow(intY, 2)));
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// use basic trigonometry(SOH CAH TOA) to calculate angle between chars
	double angleBetweenChars(const Character &firstChar, const Character &secondChar) {
		double dblAdj = abs(firstChar.intCenterX - secondChar.intCenterX);
		double dblOpp = abs(firstChar.intCenterY - secondChar.intCenterY);
		double dblAngleInRad = atan(dblOpp / dblAdj);
		double dblAngleInDeg = dblAngleInRad * (180.0 / CV_PI);
		return(dblAngleInDeg);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
	// this is to prevent including the same char twice if two contours are found for the same char,
	// for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
	std::vector<Character> removeInnerOverlappingChars(std::vector<Character> &vectorOfMatchingChars) {
		std::vector<Character> vectorOfMatchingCharsWithInnerCharRemoved(vectorOfMatchingChars);
		for (auto &currentChar : vectorOfMatchingChars) {
			for (auto &otherChar : vectorOfMatchingChars) {
				if (currentChar != otherChar) {                         // if current char and other char are not the same char . . .
																		// if current char and other char have center points at almost the same location . . .
					if (distanceBetweenChars(currentChar, otherChar) < (currentChar.dblDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY)) {
						// if we get in here we have found overlapping chars
						// next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
						// if current char is smaller than other char
						if (currentChar.boundingRect.area() < otherChar.boundingRect.area()) {
							// look for char in vector with an iterator
							std::vector<Character>::iterator currentCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), currentChar);
							// if iterator did not get to end, then the char was found in the vector
							if (currentCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
								vectorOfMatchingCharsWithInnerCharRemoved.erase(currentCharIterator);       // so remove the char
							}
						}
						else {        // else if other char is smaller than current char
									  // look for char in vector with an iterator
							std::vector<Character>::iterator otherCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), otherChar);
							// if iterator did not get to end, then the char was found in the vector
							if (otherCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
								vectorOfMatchingCharsWithInnerCharRemoved.erase(otherCharIterator);         // so remove the char
							}
						}
					}
				}
			}
		}

		return(vectorOfMatchingCharsWithInnerCharRemoved);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<std::string> recognizeCharsInDimSet(cv::Mat &imageDimSetThreshold, std::vector<Character> &vectorOfMatchingChars) {
		std::string dimNoteStrings;               // this will be the return value, the chars in the dimGDTNoteSet
		std::string accuracyString;
		std::string confidenceString;
		std::vector<std::string>vectorStrings;

		cv::Mat imageDimSetThresholdColor;
		int noOfChars = 0;

		// sort chars from left to right
		std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), Character::sortCharsLeftToRight);

		cv::cvtColor(imageDimSetThreshold, imageDimSetThresholdColor, CV_GRAY2BGR);       // make color version of threshold image so we can draw contours in color on it
		for (auto &currentChar : vectorOfMatchingChars)
		{
			noOfChars++;
		}
		for (auto &currentChar : vectorOfMatchingChars)
		{           // for each char in dimGDTNoteSet
			cv::rectangle(imageDimSetThresholdColor, currentChar.boundingRect, SCALAR_GREEN, 2);       // draw green box around the char

			cv::Mat imgROItoBeCloned = imageDimSetThreshold(currentChar.boundingRect);                 // get ROI image of bounding rect

			cv::Mat imgROI = imgROItoBeCloned.clone();      // clone ROI image so we don't change original when we resize

			cv::Mat imgROIResized;
			// resize image, this is necessary for char recognition
			cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));

			cv::Mat matROIFloat;

			imgROIResized.convertTo(matROIFloat, CV_32FC1);         // convert Mat to float, necessary for call to findNearest

			cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // flatten Matrix into one row

			cv::Mat matCurrentChar(0, 0, CV_32F);                   // declare Mat to read current char into, this is necessary b/c findNearest requires a Mat

			cv::Mat dist;
			double temp = 0;

			int iConfidence = 0;
			int iAccuracyLevel = 0;

			double confidence = 0;
			double accuracyLevel = 0;

			double totdistance = 0;
			double accurateDistance = 0;

			int k = kNearest->getDefaultK();

			cv::Mat neighborResponses(1, k, CV_32FC1);

			kNearest->findNearest(matROIFlattenedFloat, k, matCurrentChar, neighborResponses, dist);//we call find_nearest ro do the OCR!!!

			int* accuracy = new int[k];

			for (size_t i = 0; i < k; i++)
			{
				if (char(int((float)neighborResponses.at<float>(0, i))) == char(int((float)matCurrentChar.at<float>(0, 0))))
				{
					accuracy[i] = 1;
				}
				if (accuracy[i] == 1)
				{
					accuracyLevel = accuracyLevel + 1.0;
				}
			}
			for (size_t i = 0; i < k; i++)
			{
				if (dist.at<double>(0, i) < (-1 * pow(10, 50)))
				{
					dist.at<double>(0, i) = 0;
				}
				if ((dist.at<double>(0, i) > (-1)) && (dist.at<double>(0, i) < (-1)))
				{
					dist.at<double>(0, i) = 0;
				}
				temp = abs(dist.at<double>(0, i));
				totdistance = temp + totdistance;

				if (accuracy[i] == 1)
				{
					accurateDistance = temp + accurateDistance;
				}
			}

			delete[] accuracy;
			accuracy = NULL;

			confidence = (accurateDistance / totdistance) * 100;
			accuracyLevel = (accuracyLevel / k) * 100;

			iConfidence = confidence;
			iAccuracyLevel = accuracyLevel;

			accuracyString = accuracyString + char(iAccuracyLevel);
			confidenceString = confidenceString + char(iConfidence);

			float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       // convert current char from Mat to float

			dimNoteStrings = dimNoteStrings + char(int(fltCurrentChar));        // append current char to full string
		}
		vectorStrings.push_back(dimNoteStrings);
		vectorStrings.push_back(accuracyString);
		vectorStrings.push_back(confidenceString);
		return(vectorStrings);// return result
	}
	std::vector<DimSet> dimensionNoteRecognition(cv::Mat &temp)
	{
		cv::Mat tempresize;

		std::vector<DimSet> vectorOfdimGDTNoteSet;
		inputImage = temp;
		tempresize = temp.clone();

		if (inputImage.empty()) {// if unable to open image
			exit(-1);
		}

		vectorOfdimGDTNoteSet = detectDimSet(inputImage);// detect dimGDTNoteSet

		vectorOfdimGDTNoteSet = detectChars(vectorOfdimGDTNoteSet);// detect chars in dimGDTNoteSet

		if (vectorOfdimGDTNoteSet.empty())
		{
			exit(-1);
		}
		else {// if we get in here vector of possible dimGDTNoteSet has at leat one dimGDTNoteSet
			std::sort(vectorOfdimGDTNoteSet.begin(), vectorOfdimGDTNoteSet.end(), DimSet::sortDescendingByNumberOfChars);
			// suppose the dimGDTNoteSet with the most recognized chars (the first dimGDTNoteSet in sorted by string length descending order) is the actual dimGDTNoteSet
			DimSet dimensionSet = vectorOfdimGDTNoteSet.front();

			for (size_t i = 0; i < vectorOfdimGDTNoteSet.size(); i++)
			{
				DimSet tempdimensionSet = vectorOfdimGDTNoteSet.at(i);
				drawRectAroundDimSet(inputImage, tempdimensionSet);
			}

			cv::getRectSubPix(temp, dimensionSet.dimSetLocations.size, dimensionSet.dimSetLocations.center, tempresize);

			if (dimensionSet.dimNoteStrings.length() == 0) {// if no chars were found in the dimGDTNoteSet
				exit(-1);
			}

	/*		string filename = "Image_";
			if (iterIndex == 1)
			{
				filename.append("Iter_1");
			}
			if (iterIndex == 2)
			{
				filename.append("Iter_2");
			}
			if (iterIndex == 3)
			{
				filename.append("Iter_3");
			}
			if (iterIndex == 4)
			{
				filename.append("Iter_4");
			}
			filename.append(".png");
			cv::imwrite(filename, inputImage); // write iteration image out to file*/
		}
		return vectorOfdimGDTNoteSet;
	}

	inline void drawRotatedRect(cv::Mat& image, cv::RotatedRect&rRect, cv::Scalar color, bool lines)
	{
		cv::Point2f vertices2f[4];
		cv::Point vertices[4];
		rRect.points(vertices2f);
		for (int i = 0; i < 4; ++i)
		{
			vertices[i] = vertices2f[i];
		}
		if (lines == 1)
		{
			for (int i = 0; i < 4; i++)
			{                                       // draw 4 black lines
				cv::line(image, vertices2f[i], vertices2f[(i + 1) % 4], SCALAR_BLACK, 2);
			}
		}

		cv::fillConvexPoly(image, vertices, 4, color);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////
	void drawRectAroundDimSet(cv::Mat &inputImage, DimSet &dimensionSet) {
		cv::Point2f p2fRectPoints[4];
		dimensionSet.dimSetLocations.points(p2fRectPoints);            // get 4 vertices of rotated rect
		cv::Mat mask = inputImage;

		for (int i = 0; i < 4; i++)
		{                                       // draw 4 black lines
			cv::line(inputImage, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_BLACK, 1);
		}

		drawRotatedRect(inputImage, dimensionSet.dimSetLocations, SCALAR_WHITE, 1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////

	DimSet newDimNoteDimRecognition(cv::Mat &src)
	{
		std::string dimNoteStrings;               // this will be the return value, the chars in the dimGDTNoteSet
		std::string accuracyString;
		std::string confidenceString;
		std::vector<std::string>vectorStrings;
		DimSet tempDimSet;// return value
		dummyCenter.x = 1;
		dummyCenter.y = 1;
		float 	dummyAngle = 0;

		dummyDimSet.dimSetLocations = cv::RotatedRect(dummyCenter, cv::Size2f(1, 1), dummyAngle);
		dummyDimSet.dimNoteStrings = "dummy";

		std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
		std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly

		char charac = 0;

		// read in training classifications ///////////////////////////////////////////////////

		cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector

		cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

		if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
			std::wcout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
			return dummyDimSet;                                                                                  // and exit program
		}

		fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
		fsClassifications.release();                                        // close the classifications file
																			// read in training images ////////////////////////////////////////////////////////////

		cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

		cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          // open the training images file

		if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
			std::wcout << "error, unable to open training images file, exiting program\n\n";         // show error message
			return dummyDimSet;                                                                              // and exit program
		}

		fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
		fsTrainingImages.release();                                                 // close the traning images file
																					// train //////////////////////////////////////////////////////////////////////////////
		cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object
																					// finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
																					// even though in reality they are multiple images / numbers
		kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

		// test ///////////////////////////////////////////////////////////////////////////////

		cv::Mat matTestingNumbers = src;            // read in the test numbers image

		if (matTestingNumbers.empty())
		{                                // if unable to open image
			std::wcout << "error: image not read from file\n\n";         // show error message on command line
			return dummyDimSet;                                                  // and exit program
		}

		cv::Mat matGrayscale;           //
		cv::Mat matBlurred;             // declare more image variables
		cv::Mat matThresh;              //
		cv::Mat matThreshCopy;          //

		cv::cvtColor(matTestingNumbers, matGrayscale, CV_BGR2GRAY);         // convert to grayscale

																			// blur
		cv::GaussianBlur(matGrayscale,              // input image
			matBlurred,                // output image
			cv::Size(5, 5),            // smoothing window width and height in pixels
			0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

									   // filter image from grayscale to black and white
		cv::adaptiveThreshold(matBlurred,                           // input image
			matThresh,                            // output image
			255,                                  // make pixels that pass the threshold full white
			cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
			cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
			11,                                   // size of a pixel neighborhood used to calculate threshold value
			2);                                   // constant subtracted from the mean or weighted mean

		matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image

		std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
		std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

		cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
			ptContours,                             // output contours
			v4iHierarchy,                           // output hierarchy
			cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
			cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

		for (int i = 0; i < ptContours.size(); i++) {               // for each contour
			ContourWithData contourWithData;                                                    // instantiate a contour with data object
			contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
			contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
			contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
			allContoursWithData.push_back(contourWithData);                                     // add contour with data object to list of all contours with data
		}

		for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
			if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
				validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
			}
		}
		// sort contours from left to right
		std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

		std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program

		for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour

			cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect

			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

			cv::Mat matROIFloat;
			matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

			cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

			cv::Mat matCurrentChar(0, 0, CV_32F);

			cv::Mat dist;
			double temp = 0;

			int iConfidence = 0;
			int iAccuracyLevel = 0;

			double confidence = 0;
			double accuracyLevel = 0;

			double totdistance = 0;
			double accurateDistance = 0;
			int k = kNearest->getDefaultK();
			cv::Mat neighborResponses(1, k, CV_32FC1);
			int* accuracy = new int[k];

			kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar, neighborResponses, dist);     // finally we can call find_nearest !!!

			for (size_t i = 0; i < k; i++)
			{
				if (char(int((float)neighborResponses.at<float>(0, i))) == char(int((float)matCurrentChar.at<float>(0, 0))))
				{
					accuracy[i] = 1;
				}
				if (accuracy[i] == 1)
				{
					accuracyLevel = accuracyLevel + 1.0;
				}
			}
			for (size_t i = 0; i < k; i++)
			{
				if (dist.at<double>(0, i) < (-1 * pow(10, 50)))
				{
					dist.at<double>(0, i) = 0;
				}
				if ((dist.at<double>(0, i) > (-1)) && (dist.at<double>(0, i) < (-1)))
				{
					dist.at<double>(0, i) = 0;
				}
				temp = abs(dist.at<double>(0, i));
				totdistance = temp + totdistance;

				if (accuracy[i] == 1)
				{
					accurateDistance = temp + accurateDistance;
				}

			}

			delete[] accuracy;
			accuracy = NULL;

			confidence = (accurateDistance / totdistance) * 100;
			accuracyLevel = (accuracyLevel / k) * 100;

			//	cout << "\nConfidence Level %:\n" << confidence<<endl;
			//	cout << "Accuracy Level %:\n" << accuracyLevel << endl;

			iConfidence = confidence;
			iAccuracyLevel = accuracyLevel;

			accuracyString = accuracyString + char(iAccuracyLevel);
			confidenceString = confidenceString + char(iConfidence);

			float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

			charac = char(int(fltCurrentChar));

			strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
		}

		tempDimSet.dimNoteStrings = strFinalString;
		tempDimSet.accuracyPercentage = accuracyString;
		tempDimSet.confidencePercentage = confidenceString;
		return tempDimSet;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	DimSet deskew_Recognize(cv::Mat toRotImg)
	{
		double angle = compute_skew(toRotImg);
		Point Center = { toRotImg.size().width / 2,toRotImg.size().height / 2 };
		cv::Mat M = getRotationMatrix2D(Center, angle, 1.0);
		double angle90 = check_90deg(toRotImg);

		if (angle90 > 60)
		{
			transpose(toRotImg, toRotImg);
			flip(toRotImg, toRotImg, 1);
		}
		else
		{
			cv::warpAffine(toRotImg, toRotImg, M, Size((1.5 * toRotImg.rows), (1.5 * toRotImg.cols)), cv::INTER_CUBIC, cv::BORDER_CONSTANT, SCALAR_WHITE);
			cv::getRectSubPix(toRotImg, toRotImg.size(), Center, toRotImg);
		}

		DimSet tempDimSet = newDimNoteDimRecognition(toRotImg);

		return tempDimSet;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<DimSet> dimNoteDimRecognition(char* ipFilename, int knn, int horVerMode)
	{
		char* inputFilename = ipFilename;

		Mat bw;
		//std::cout << "\n Enter k for kNN\n " << endl;
		//cin >> knn;

	/*	MessageBox(NULL, L"Please open in the following dialog box, the drawing for which\n dimensions, tolerancing and notes are to be recognized \n\n Supported formats:*.bmp,*.dib,*.jpeg,*.jpg,*.jpe,*.jp2,*.png,\n*.webp,*.pbm,*.pgm,*.ppm,*.sr,*.ras,*.tiff, *.tif", L"Open Drawing", MB_OK);

		HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED |
			COINIT_DISABLE_OLE1DDE);
		if (SUCCEEDED(hr))
		{
			IFileOpenDialog *pFileOpen;

			// Create the FileOpenDialog object.
			hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
				IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

			if (SUCCEEDED(hr))
			{
				// Show the Open dialog box.
				hr = pFileOpen->Show(NULL);

				// Get the file name from the dialog box.
				if (SUCCEEDED(hr))
				{
					IShellItem *pItem;
					hr = pFileOpen->GetResult(&pItem);
					if (SUCCEEDED(hr))
					{
						PWSTR pszFilePath;
						hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

						// Display the file name to the user.
						if (SUCCEEDED(hr))
						{
							WideCharToMultiByte(CP_ACP, 0, pszFilePath, -1, inputFilename, sizeof(inputFilename), NULL, NULL);

							CoTaskMemFree(pszFilePath);
						}
						pItem->Release();
					}
				}
				pFileOpen->Release();
			}
			CoUninitialize();
		}

	//	std::cout << "\n Enter horVerMode:\n0 for only horizontal\n1 for only horizontal and vertical dimensions\n2 for all orientations based search \n" << endl;
	//	cin >> horVerMode;
	//	std::cout << "\n Enter k for kNN based search \n" << endl;
	//	cin >> knn;
	*/
		std::vector<DimSet> opVectorDimsets;
		kNN = knn;
		int horizontalVerticalMode = horVerMode;

		bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();           // attempt KNN training

		if (blnKNNTrainingSuccessful == false)
		{                            // if KNN training was not successful show error message
			std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
		}

		cv::Mat temp;
		cv::Mat temprot;
		cv::Mat temp1;
		cv::Mat temprot1;
		cv::Mat iter4;

		// Load the image
		Mat src = imread(inputFilename);
		Mat srctemp = src.clone();
		// Check if image is loaded fine
		if (!src.data)
			cerr << "Problem loading image!!!" << endl;

		temp1 = lineCleanse(srctemp);

		temp1 = maximizeContrast(temp1);
		int dstWidth = temp1.cols;
		int dstHeight = temp1.rows * 2;

		dst = cv::Mat(dstHeight, dstWidth, CV_8UC3, cv::Scalar(0, 0, 0));

		temp = temp1.clone();
		trialImage = temp;

		int dummyLocations[3] = { 0,0,0 };
		dummyCenter.x = 1;
		dummyCenter.y = 1;
		float	dummyAngle = 0;

		dummyDimSet.dimSetLocations = cv::RotatedRect(dummyCenter, cv::Size2f(1, 1), dummyAngle);
		dummyDimSet.dimNoteStrings = "dummy";
		///////////////////////////////////////////////////////////////////////////////////////////////////
		//                                        Iteration 1:                                       //
		///////////////////////////////////////////////////////////////////////////////////////////////////
		if (horizontalVerticalMode >= 0)
		{
			iterIndex = 1;
			trialImageWidth = temp.size().width;
			trialImageHeight = temp.size().height;
			output.vectorOfdimGDTNoteSet1 = dimensionNoteRecognition(temp);
			opVectorDimsets.insert(opVectorDimsets.begin(), output.vectorOfdimGDTNoteSet1.begin(), output.vectorOfdimGDTNoteSet1.end());
			opVectorDimsets.push_back(dummyDimSet);

		}

		dummyLocations[0] = opVectorDimsets.size();

		temprot1 = inputImage;
		transpose(temprot1, temprot);
		flip(temprot, temprot, 1);

		for (size_t i = 0; i < dummyLocations[0]; i++)
		{
			cv::Rect boundingRect = opVectorDimsets[i].dimSetLocations.boundingRect();
			for (size_t j = 0; j < dummyLocations[0]; j++)
			{
				cv::Rect boundingRect2 = opVectorDimsets[j].dimSetLocations.boundingRect();
				bool tlContainer = boundingRect.contains(boundingRect2.tl());
				bool brContainer = boundingRect.contains(boundingRect2.br());

				if (tlContainer*brContainer)
				{
					opVectorDimsets[j].dimSetLocations = dummyDimSet.dimSetLocations;
					opVectorDimsets[j].dimNoteStrings = "dummy";
				}
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////
		//                                        Iteration 2:                                       //
		///////////////////////////////////////////////////////////////////////////////////////////////////
		inputImage = temp1;

		if (horizontalVerticalMode > 1)
		{
			iterIndex = 2;
			trialImageWidth = inputImage.size().width;
			trialImageHeight = inputImage.size().height;
			output.vectorOfdimGDTNoteSet3 = dimensionNoteRecognition(inputImage);
			opVectorDimsets.insert(opVectorDimsets.end(), output.vectorOfdimGDTNoteSet3.begin(), output.vectorOfdimGDTNoteSet3.end());
			opVectorDimsets.push_back(dummyDimSet);
		}

		dummyLocations[1] = opVectorDimsets.size();

		iter4 = inputImage;

		for (size_t i = dummyLocations[0]; i < dummyLocations[1]; i++)
		{
			cv::Rect boundingRect = opVectorDimsets[i].dimSetLocations.boundingRect();
			for (size_t j = 0; j < dummyLocations[0]; j++)
			{
				cv::Rect boundingRect2 = opVectorDimsets[j].dimSetLocations.boundingRect();
				bool tlContainer = boundingRect.contains(boundingRect2.tl());
				bool brContainer = boundingRect.contains(boundingRect2.br());

				if (tlContainer*brContainer)
				{
					opVectorDimsets[j].dimSetLocations = dummyDimSet.dimSetLocations;
					opVectorDimsets[j].dimNoteStrings = "dummy";
				}
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////
		//                                        Iteration 3:                                       //
		///////////////////////////////////////////////////////////////////////////////////////////////////
		if (horizontalVerticalMode > 0)
		{
			iterIndex = 3;
			trialImageWidth = temprot.size().width;
			trialImageHeight = temprot.size().height;
			output.vectorOfdimGDTNoteSet2 = dimensionNoteRecognition(temprot);
			opVectorDimsets.insert(opVectorDimsets.end(), output.vectorOfdimGDTNoteSet2.begin(), output.vectorOfdimGDTNoteSet2.end());
			opVectorDimsets.push_back(dummyDimSet);
		}
		dummyLocations[2] = opVectorDimsets.size();

		for (size_t i = dummyLocations[1]; i < dummyLocations[2]; i++)
		{
			cv::Rect boundingRect = opVectorDimsets[i].dimSetLocations.boundingRect();
			for (size_t j = 0; j < dummyLocations[0]; j++)
			{
				cv::Rect boundingRect2 = opVectorDimsets[j].dimSetLocations.boundingRect();
				bool tlContainer = boundingRect.contains(boundingRect2.tl());
				bool brContainer = boundingRect.contains(boundingRect2.br());

				if (tlContainer*brContainer)
				{
					opVectorDimsets[j].dimSetLocations = dummyDimSet.dimSetLocations;
					opVectorDimsets[j].dimNoteStrings = "dummy";
				}
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////
		//                                        Iteration 4:                                       //
		///////////////////////////////////////////////////////////////////////////////////////////////////
		transpose(iter4, iter4);
		flip(iter4, iter4, 1);
		iter4 = lineCleanse(iter4);

		if (horizontalVerticalMode > 1)
		{
			iterIndex = 4;
			trialImageWidth = iter4.size().width;
			trialImageHeight = iter4.size().height;
			output.vectorOfdimGDTNoteSet4 = dimensionNoteRecognition(iter4);
			opVectorDimsets.insert(opVectorDimsets.end(), output.vectorOfdimGDTNoteSet4.begin(), output.vectorOfdimGDTNoteSet4.end());
		}
		for (size_t i = dummyLocations[2]; i < opVectorDimsets.size(); i++)
		{
			cv::Rect boundingRect = opVectorDimsets[i].dimSetLocations.boundingRect();
			for (size_t j = 0; j < dummyLocations[0]; j++)
			{
				cv::Rect boundingRect2 = opVectorDimsets[j].dimSetLocations.boundingRect();
				bool tlContainer = boundingRect.contains(boundingRect2.tl());
				bool brContainer = boundingRect.contains(boundingRect2.br());

				if (tlContainer*brContainer)
				{
					opVectorDimsets[j].dimSetLocations = dummyDimSet.dimSetLocations;
					opVectorDimsets[j].dimNoteStrings = "dummy";
				}
			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////////
		//                                        Updating rotated rectangles:                           //
		///////////////////////////////////////////////////////////////////////////////////////////////////
		std::vector<cv::RotatedRect> rotRects;

		for (size_t i = dummyLocations[1]; i < opVectorDimsets.size(); i++)
		{
			cv::Point2f updatedCenter;

			double swap = opVectorDimsets[i].dimSetLocations.center.x;
			updatedCenter.x = opVectorDimsets[i].dimSetLocations.center.y;
			updatedCenter.y = trialImage.size().height - swap;

			cv::RotatedRect tempRect = cv::RotatedRect(updatedCenter, cv::Size2f(opVectorDimsets[i].dimSetLocations.size.height, opVectorDimsets[i].dimSetLocations.size.width), (opVectorDimsets[i].dimSetLocations.angle - 180));
			opVectorDimsets[i].dimSetLocations = tempRect;

		}

		for (size_t i = 0; i < opVectorDimsets.size(); i++)
		{
			drawRotatedRect(trialImage, opVectorDimsets[i].dimSetLocations, SCALAR_WHITE, 1);
		}

		std::string opfilename = inputFilename;
		opfilename[(opfilename.size() - 4)] = '_';
		opfilename.append("_Output.jpg");
		cv::imwrite(opfilename, trialImage);
		///////////////////////////////////////////////////////////////////////////////////////////////////
		//                                        Post Processing:                                       //
		///////////////////////////////////////////////////////////////////////////////////////////////////
		cv::Point2f completeWhiteImageCenter;
		cv::Mat BWtrialImage;
		completeWhiteImageCenter.x = trialImage.size().width / 2;
		completeWhiteImageCenter.y = trialImage.size().height / 2;

		cv::RotatedRect completeWhiteImage = cv::RotatedRect(completeWhiteImageCenter, cv::Size2f(trialImage.size().width, trialImage.size().height), 0);

		drawRotatedRect(trialImage, completeWhiteImage, SCALAR_WHITE, 0);

		for (size_t i = 0; i < opVectorDimsets.size(); i++)
		{
			drawRectAroundDimSet(trialImage, opVectorDimsets[i]);
		}
		cv::imwrite("OutputDimensions.jpg", trialImage);

		preprocess(trialImage, BWtrialImage, BWtrialImage);        // preprocess to get grayscale and threshold images																						
		cv::threshold(BWtrialImage, BWtrialImage, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU); 	// threshold again to eliminate any gray areas

		std::vector<std::vector<cv::Point> > contours;
		vector<Vec4i> hierarchy;

		cv::findContours(BWtrialImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

		cv::Mat imgWhite2(trialImage.size(), trialImage.type(), SCALAR_WHITE);
		cv::Mat imgBlack(trialImage.size(), trialImage.type(), SCALAR_BLACK);
		int finalContourSize = 0;

		cv::Point dummyCounterPt1;
		cv::Point dummyCounterPt2;
		dummyCounterPt1.x = 0;
		dummyCounterPt1.y = 0;
		dummyCounterPt2.x = 1;
		dummyCounterPt2.y = 1;
		std::vector<cv::Point> dummyContourPoints;
		std::vector<cv::RotatedRect> finContourRotRecs;
		std::vector<cv::Rect> finContourRecs;

		dummyContourPoints.push_back(dummyCounterPt1);
		dummyContourPoints.push_back(dummyCounterPt2);

		for (unsigned int i = 0; i < contours.size(); i++)
		{
			if (hierarchy[i][3] == -1)
			{
				int noOfLines;
				finalContourSize++;
				cv::RotatedRect boundRotRect = minAreaRect(contours[i]);
				cv::Rect boundRect = boundingRect(contours[i]);
				finContourRotRecs.push_back(boundRotRect);
				finContourRecs.push_back(boundRect);
			}
			else
			{
				contours[i] = dummyContourPoints;
			}
		}
		for (unsigned int i = 0; i < finContourRotRecs.size(); i++)
		{
			drawRotatedRect(imgWhite2, finContourRotRecs[i], SCALAR_GREEN, 0);
			rectangle(imgWhite2, finContourRecs[i], SCALAR_RED, 2);
		}
		double alpha = 0.5;
		double beta = (1.0 - alpha);
		addWeighted(imgWhite2, alpha, srctemp, beta, 0.0, srctemp);
		srctemp = maximizeContrast(srctemp);
		cv::imwrite("PostProcessed.jpg", srctemp);
		bw.release();
		///////////////////////////////////////////////////////////////////////////////////////////////////
		//                                        User Input:                                            //
		///////////////////////////////////////////////////////////////////////////////////////////////////
		Mat frame = srctemp;
		namedWindow("Select ROI", WINDOW_KEEPRATIO);
		setMouseCallback("Select ROI", onMouse, 0);
	
		for (;;)
		{
			frame.copyTo(image);

			if (insert && selection.width > 0 && selection.height > 0)
			{
				rectangle(image, Point(selection.x, selection.y), Point(selection.x + selection.width, selection.y + selection.height), CV_RGB(255, 255, 255));
			}

			imshow("Select ROI", image);

			char k = waitKey(0);

			if (k == 27)//Escape key breaks the loop and ends
			{
				cv::destroyAllWindows;
				break;
			}
			if (k == 'x' || k == 'X')// x key shows selected images
			{
				cout << "\n\n Size of setOfnewdimGDTNoteSet:\n " << setOfnewdimGDTNoteSet.size() << endl;
				for (int i = 0; i < setOfnewdimGDTNoteSet.size(); i++)
				{
					if (setOfnewdimGDTNoteSet[i].height != 0)
					{
						cv::Mat newDimSetImgRotCrop = deskew(setOfnewdimGDTNoteSet[i].newDimSetImg);
						if (setOfnewdimGDTNoteSet.size() > 0)
						{
							DimSet tempDimSet = newDimNoteDimRecognition(newDimSetImgRotCrop);
							opVectorDimsets.push_back(tempDimSet);
							cout << "\n\nThe new strings:\n" << tempDimSet.dimNoteStrings << endl;
							waitKey(0);
						}
						else
						{
							break;
						}
						cv::destroyAllWindows;
					}
				}
			}
			else if (k == 'i' || k == 'I')//I key enables selection
			{
				insert = !insert;
			}
		}
		Mat tempRotRects;
		std::vector<DimSet> tempOutput;
		std::vector<DimSet> finalOutput;
		DimSet tempDimSet1;
		DimSet tempDimSet2;

		for (size_t i = 0; i < finContourRotRecs.size(); i++)
		{
			Size rect_size = finContourRotRecs[i].size;
			Mat M, rotated;

			double angle = finContourRotRecs[i].angle;

			if (angle < -45.0) {
				angle += 90.0;
				swap(rect_size.width, rect_size.height);
			}
			// get the rotation matrix
			M = getRotationMatrix2D(finContourRotRecs[i].center, angle, 1.0);
			// perform the affine transformation
			cv::warpAffine(srctemp, rotated, M, Size((1.5 * srctemp.rows), (1.5 * srctemp.cols)), cv::INTER_CUBIC, cv::BORDER_CONSTANT, SCALAR_WHITE);
			// crop the resulting image
			getRectSubPix(rotated, rect_size, finContourRotRecs[i].center, tempRotRects);
			cvtColor(tempRotRects, tempRotRects, CV_BGR2GRAY);
			angle = check_90deg(tempRotRects);
			cvtColor(tempRotRects, tempRotRects, CV_GRAY2BGR);

			Mat clockRot = tempRotRects;
			Mat antiClockRot = tempRotRects;

			if (angle > 45)
			{
				transpose(clockRot, clockRot);
				flip(clockRot, clockRot, 1);
				transpose(antiClockRot, antiClockRot);
				flip(antiClockRot, antiClockRot, -1);
			}
			tempDimSet1.imageDimSet = clockRot;
			tempDimSet2.imageDimSet = antiClockRot;
			tempOutput.push_back(tempDimSet1);
			tempOutput.push_back(tempDimSet2);
		}
		cv::destroyAllWindows;

		for (size_t i = 0; i < tempOutput.size(); i++)
		{
			if (tempOutput[i].dimSetLocations.size.width != 1)
			{
				if (tempOutput[i].dimNoteStrings[0] != 'd')
				{
					if (tempOutput[i].imageDimSet.empty() != TRUE)
					{
						finalOutput.push_back(tempOutput[i]);
					}
				}
			}
		}

		char k = waitKey(0);

		if (k == 27)//Escape key breaks the loop and ends
		{
			cv::destroyAllWindows;
		}

		return finalOutput;

	}
}