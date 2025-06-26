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

namespace dimNoteDimRecog
{
	opVector output;
	int iterIndex;
	int intdimOrNoteSetCounter = 0;
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
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	Point pos;

	Rect PatchRect;
	Mat PatchImg;
	const int RESIZED_IMAGE_WIDTH = 20;
	const int RESIZED_IMAGE_HEIGHT = 30;
	cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();

	class newDimSet
	{
	public:
		Mat newDimSetImg;
		double x;
		double y;
		double width;
		double height;
	};
	std::vector <newDimSet> setOfnewDimSets;

	class ContourWithData {
	public:
		// member variables ///////////////////////////////////////////////////////////////////////////
		std::vector<cv::Point> ptContour;           // contour
		cv::Rect boundingRect;                      // bounding rect for contour
		float fltArea;                              // area of contour

													///////////////////////////////////////////////////////////////////////////////////////////////
		bool checkIfContourIsValid() {                              // obviously in a production grade program
			if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for 
			return true;                                            // identifying if a contour is valid !!
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
			return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
		}

	};

	double getOrientation(vector<Point> &pts, Mat &img)
	{
		if (pts.size() == 0) return false;
		//Construct a buffer used by the pca analysis
		Mat data_pts = Mat(pts.size(), 2, CV_64FC1);
		for (int i = 0; i < data_pts.rows; ++i)
		{
			data_pts.at<double>(i, 0) = pts[i].x;
			data_pts.at<double>(i, 1) = pts[i].y;
		}
		//Perform PCA analysis
		PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
		//Store the position of the object
		pos = Point(pca_analysis.mean.at<double>(0, 0),
			pca_analysis.mean.at<double>(0, 1));

		for (int i = 0; i < 2; ++i) //Store the eigenvalues and eigenvectors
		{
			eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
				pca_analysis.eigenvectors.at<double>(i, 1));

			eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
		}

		return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
	}

	double GetMode(double daArray[], int iSize) {
		// Allocate an int array of the same size to hold the
		// repetition count
		int* ipRepetition = new int[iSize];
		for (int i = 0; i < iSize; ++i) {
			ipRepetition[i] = 0;
			int j = 0;
			bool bFound = false;
			while ((j < i) && (daArray[i] != daArray[j]))
			{
				if (daArray[i] != daArray[j])
				{
					++j;
				}
			}
			++(ipRepetition[j]);
		}

		int iMaxRepeat = 0;
		for (int i = 1; i < iSize; ++i)
		{
			if (daArray[i] != 0)
			{
				if (ipRepetition[i] >= ipRepetition[iMaxRepeat])
				{
					iMaxRepeat = i;
				}
			}

		}

		delete[] ipRepetition;
		ipRepetition = NULL;
		return daArray[iMaxRepeat];
	}

	cv::Mat RotToHor(Mat bw)
	{
		double EigVecX = 0;
		double EigVecY = 0;
		Mat dst, fdst;

		cvtColor(bw, bw, COLOR_BGR2GRAY);
	//	threshold(bw, bw, 150, 255, CV_THRESH_BINARY);

		double AngleDeg = 0;

		int validcontours = 0;
		double area = 0;

		int PosMajorDirX = 0;
		int NegMajorDirX = 0;

		int ChoiceMajorDirX = 0;

		float tempX = 0;
		float tempY = 0;

		double mode = 0;
		int ConsideredContours = 0;

		Point2f pt(bw.cols / 2., bw.rows / 2.);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		//bitwise_not(bw, bw);

		findContours(bw, contours, hierarchy, RETR_TREE, CV_CHAIN_APPROX_NONE);

		// For each object
		for (size_t i = 0; i < contours.size(); ++i)
		{
			if (hierarchy[i][3] == -1)
			{
				drawContours(bw, contours, i, SCALAR_WHITE, 3, 8, hierarchy, 0);
			}
			cv::imshow("White border Image", bw);
			waitKey(0);
			// Calculate its area
			area = contourArea(contours[i]);
			validcontours++;
		}
	
	/*	for (size_t i = 0; i < contours.size(); ++i)
		{
			if (hierarchy[i][3] != -1)
			{
				drawContours(bw, contours, i, SCALAR_BLACK, 3, 8, hierarchy, 0);
			}
			cv::imshow("enhanced black drawn Image", bw);
			waitKey(0);
		}
		
		cv::waitKey(0);*/

		Mat temp;
		int q = 0;

		double** EigVecs = new double*[contours.size()];
		for (int i = 0; i < contours.size(); ++i)
		{
			EigVecs[i] = new double[2];
			for (size_t j = 0; j < 2; j++)
			{
				EigVecs[i][j] = 0;
			}
		}
		double** RoundEigVecs = new double*[contours.size()];
		for (int i = 0; i < contours.size(); ++i)
		{
			RoundEigVecs[i] = new double[2];
			for (size_t j = 0; j < 2; j++)
			{
				RoundEigVecs[i][j] = 0;
			}
		}
		int* ValContours = new int[contours.size()];

		for (size_t i = 0; i < contours.size(); i++)
		{
			ValContours[i] = 0;
		}
		double* RValAngles = new double[contours.size()];
		for (size_t i = 0; i < contours.size(); i++)
		{
			RValAngles[i] = 0;
		}

		for (size_t i = 0; i < contours.size(); ++i)
		{
			// Calculate its area
			area = contourArea(contours[i]);
		
			ValContours[i] = i;
			// Get the object orientation
			double notusedangle = getOrientation(contours[i], bw);

			double temp;
			temp = abs(10 * eigen_vecs[0].x);
			double sign = eigen_vecs[0].x / abs(eigen_vecs[0].x);
			double frac = temp - floor(temp);

			if (frac >= .5)
			{
				temp = ceil(temp);
			}
			else
			{
				temp = floor(temp);
			}

			double REigX = (temp / 10)*sign;

			temp = abs(10 * eigen_vecs[0].y);
			sign = eigen_vecs[0].y / abs(eigen_vecs[0].y);
			frac = temp - floor(temp);

			if (frac >= .5)
			{
				temp = ceil(temp);
			}
			else
			{
				temp = floor(temp);
			}

			double REigY = (temp / 10)*sign;

			EigVecs[i][0] = eigen_vecs[0].x;
			EigVecs[i][1] = eigen_vecs[0].y;

			RoundEigVecs[i][0] = REigX;
			RoundEigVecs[i][1] = REigY;

			if ((eigen_vecs[0].x == 1) || (eigen_vecs[0].x == -1) || (eigen_vecs[0].y == 1) || (eigen_vecs[0].y == -1)) continue;
			ValContours[i] = i;
			ConsideredContours++;
			RValAngles[i] = ((atan2(RoundEigVecs[i][1], RoundEigVecs[i][0]) / 3.141592) * 180);
		}

		mode = GetMode(RValAngles, contours.size());

		for (size_t i = 0; i < contours.size(); ++i)
		{
			if (mode == RValAngles[i])
			{
				ValContours[i] = i;
			}
			else
			{
				ValContours[i] = 0;
			}
		}
		validcontours = 0;

		for (size_t i = 0; i < contours.size(); ++i)
		{
			if (ValContours[i] != 0)
			{
				EigVecX = EigVecX + EigVecs[i][0];
				EigVecY = EigVecY + EigVecs[i][1];
				validcontours++;
			}
		}

		eigen_vecs[0].x = EigVecX / validcontours;
		eigen_vecs[0].y = EigVecY / validcontours;
		eigen_vecs[1].x = 0;
		eigen_vecs[1].y = 0;

		EigVecX = eigen_vecs[0].x;
		EigVecY = eigen_vecs[0].y;

	/*	cv::Mat top_left
			= bw(cv::Range(0, bw.rows / 2 - 1), cv::Range(0, bw.cols / 2 - 1));
		cv::Mat top_right
			= bw(cv::Range(0, bw.rows / 2 - 1), cv::Range(bw.cols / 2, bw.cols - 1));
		cv::Mat bottom_left
			= bw(cv::Range(bw.rows / 2, bw.rows - 1), cv::Range(0, bw.cols / 2 - 1));
		cv::Mat bottom_right
			= bw(cv::Range(bw.rows / 2, bw.rows - 1), cv::Range(bw.cols / 2, bw.cols - 1));

		int Inttop_left = 0;
		int Inttop_right = 0;
		int Intbottom_left = 0;
		int Intbottom_right = 0;
		int tempInt = 0;

		int** white = new int*[bw.rows];
		for (int i = 0; i < bw.rows; ++i)
			white[i] = new int[bw.cols];

		for (size_t i = 0; i < bw.rows; i++)
		{
			for (size_t j = 0; j < bw.cols; j++)
			{
				white[i][j] = 255;
			}
		}

		for (size_t i = 0; i < top_left.size().width; i++)
		{
			for (size_t j = 0; j < top_left.size().height; j++)
			{
				tempInt = top_left.at<unsigned char>(i, j) - white[i][j];
				Inttop_left = Inttop_left + tempInt;
			}
		}

		for (size_t i = 0; i < top_right.size().width; i++)
		{
			for (size_t j = 0; j < top_right.size().height; j++)
			{
				tempInt = top_right.at<unsigned char>(i, j) - white[i][j];
				Inttop_right = Inttop_right + tempInt;
			}
		}

		for (size_t i = 0; i < bottom_left.size().width; i++)
		{
			for (size_t j = 0; j < bottom_left.size().height; j++)
			{
				tempInt = bottom_left.at<unsigned char>(i, j) - white[i][j];
				Intbottom_left = Intbottom_left + tempInt;
			}
		}

		for (size_t i = 0; i < bottom_right.size().width; i++)
		{
			for (size_t j = 0; j < bottom_right.size().height; j++)
			{
				tempInt = bottom_right.at<unsigned char>(i, j) - white[i][j];
				Intbottom_right = Intbottom_right + tempInt;
			}
		}

		Inttop_left = abs(Inttop_left);
		Inttop_right = abs(Inttop_right);
		Intbottom_left = abs(Intbottom_left);
		Intbottom_right = abs(Intbottom_right);

		int QuadIntensity[4] = { Inttop_left ,Inttop_right,Intbottom_left,Intbottom_right };

		int MaxQuadIntensity = *std::max_element(QuadIntensity, QuadIntensity + 4);

		if (MaxQuadIntensity == QuadIntensity[0])
		{
			AngleDeg = ((atan2((-1 * abs(EigVecY)), abs(EigVecX)) / 3.141592) * 180) - 90;
		}

		if (MaxQuadIntensity == QuadIntensity[1])
		{
			AngleDeg = ((atan2(abs(EigVecY), abs(EigVecX)) / 3.141592) * 180) - 90;
		}

		if (MaxQuadIntensity == QuadIntensity[2])
		{
			AngleDeg = ((atan2(abs(EigVecY), abs(EigVecX)) / 3.141592) * 180) + 90;
		}

		if (MaxQuadIntensity == QuadIntensity[3])
		{
			AngleDeg = ((atan2((-1 * abs(EigVecY)), abs(EigVecX)) / 3.141592) * 180) + 90;
		}*/

		AngleDeg = ((atan2(( abs(EigVecY)), abs(EigVecX)) / 3.141592) * 180);
		Mat M = getRotationMatrix2D(pt, AngleDeg, 1.0);

		warpAffine(bw, dst, M, Size((3 * bw.rows), (3 * bw.cols)));

	/*	for (int i = 0; i < 2; i++) {
			delete[] EigVecs[i];
		}
		delete[] EigVecs;*/
		EigVecs = NULL;
	/*	for (int i = 0; i < 2; i++)
			delete[] RoundEigVecs[i];
		delete[] RoundEigVecs;*/
		RoundEigVecs = NULL;
	//	delete[] ValContours;
		ValContours = NULL;
	/*	for (int i = 0; i < bw.rows; ++i) {
			delete[] white[i];
		}*/
		//white= NULL;
		return dst;
		namedWindow("Rotated Image", WINDOW_KEEPRATIO);
		cv::imshow("Rotated Image", dst);
	}

	static void onMouse(int event, int x, int y, int, void*)
	{
		newDimSet newDimNoteSet;
		newDimNoteSet.x = 0;
		newDimNoteSet.y = 0;
		newDimNoteSet.width = 0;
		newDimNoteSet.newDimSetImg = Mat(image, Rect(0, 0, 0, 0));

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
			cv::destroyWindow("Selected Img");
		case CV_EVENT_LBUTTONUP:
			if (selectObject && insert)
			{
				if (selection.width > 5 && selection.height > 5)
				{
					newDimNoteSet.x = selection.x;
					newDimNoteSet.y = selection.y;
					newDimNoteSet.width = selection.width;
					newDimNoteSet.height = selection.height;
					PatchRect = selection;
					image(PatchRect).copyTo(PatchImg);
					imshow("Selected Img", PatchImg);
					newDimNoteSet.newDimSetImg = PatchImg;
					setOfnewDimSets.push_back(newDimNoteSet);
				}
				else
					selection = Rect(0, 0, 0, 0);
			}
			selectObject = false;
			insert = false;
			break;
		}
	}

	Character::Character(std::vector<cv::Point> _contour) {
		contour = _contour;

		boundingRect = cv::boundingRect(contour);

		intCenterX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
		intCenterY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

		dblDiagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));

		dblAspectRatio = (float)boundingRect.width / (float)boundingRect.height;
	}

	cv::Mat maximizeContrast(cv::Mat &imageDimSetGSBW) {
		cv::Mat imgTopHat;
		cv::Mat imgBlackHat;
		cv::Mat imageDimSetGSBWPlusTopHat;
		cv::Mat imageDimSetGSBWPlusTopHatMinusBlackHat;

		cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));

		cv::morphologyEx(imageDimSetGSBW, imgTopHat, CV_MOP_TOPHAT, structuringElement);
		cv::morphologyEx(imageDimSetGSBW, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

		imageDimSetGSBWPlusTopHat = imageDimSetGSBW + imgTopHat;
		imageDimSetGSBWPlusTopHatMinusBlackHat = imageDimSetGSBWPlusTopHat - imgBlackHat;

		return(imageDimSetGSBWPlusTopHatMinusBlackHat);
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

																										// call adaptive threshold to get imageDimSetThreshold
																										//threshold(imgBlurred, imageDimSetThreshold, 150, 255, CV_THRESH_BINARY);
		cv::adaptiveThreshold(imgBlurred, imageDimSetThreshold, 255.0, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//                                        Detect Dimensions or Notes:                            //
	///////////////////////////////////////////////////////////////////////////////////////////////////

	std::vector<DimSet> detectDimSet(cv::Mat &inputImage) {
		std::vector<DimSet> vectorOfDimSets;			// this will be the return value

		cv::Mat imageDimSetGSBWScene;
		cv::Mat imageDimSetThresholdScene;
		cv::RNG rng;

		cv::destroyAllWindows();

		preprocess(inputImage, imageDimSetGSBWScene, imageDimSetThresholdScene);        // preprocess to get grayscale and threshold images
																						// find all possible chars in the scene,
																						// this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
		std::vector<Character> vectorOfCharactersInScene = findCharactersFromInput(imageDimSetThresholdScene);
		std::vector<std::vector<cv::Point> > contours;

		for (auto &Character : vectorOfCharactersInScene) {
			contours.push_back(Character.contour);
		}
		// given a vector of all possible chars, find groups of matching chars
		// in the next steps each group of matching chars will attempt to be recognized as a dimOrNoteSet
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
			DimSet posDimSet = extractDimSet(inputImage, vectorOfMatchingChars);        // attempt to extract dimOrNoteSet

			if (posDimSet.imageDimSet.empty() == false) {                                              // if dimOrNoteSet was found
				vectorOfDimSets.push_back(posDimSet);                                        // add to vector of possible dimOrNoteSets
			}
		}
		//std::cout << std::endl << vectorOfDimSets.size() << " Possible dimOrNoteSets found" << std::endl;    	
		for (unsigned int i = 0; i < vectorOfDimSets.size(); i++) {
			cv::Point2f p2fRectPoints[4];
			vectorOfDimSets[i].dimSetLocations.points(p2fRectPoints);
		}
		//std::cout << std::endl << "dimOrNoteSet detection complete, click on any image to begin char recognition . . ." << std::endl << std::endl;
		return vectorOfDimSets;
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

		// calculate the center point of the dimOrNoteSet
		double dbldimOrNoteSetCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
		double dbldimOrNoteSetCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;
		cv::Point2d p2ddimOrNoteSetCenter(dbldimOrNoteSetCenterX, dbldimOrNoteSetCenterY);
		// calculate dimOrNoteSet width and height
		int intdimOrNoteSetWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * dimOrNoteSet_WIDTH_PADDING_FACTOR);

		double intTotalOfCharHeights = 0;
	
		for (int i = 0; i < vectorOfMatchingChars.size(); i++)
		{
			intTotalOfCharHeights = intTotalOfCharHeights + vectorOfMatchingChars[i].boundingRect.height;		
		}

		double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();

		int intdimOrNoteSetHeight = (int)(dblAverageCharHeight * dimOrNoteSet_HEIGHT_PADDING_FACTOR);
		// calculate correction angle of dimOrNoteSet region
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
		// assign rotated rect member variable of possible dimOrNoteSet

		int intdimOrNoteSetArea = intdimOrNoteSetHeight*intdimOrNoteSetWidth;

		if (intdimOrNoteSetArea < ((trialImageWidth*trialImageHeight) / 16))
		{
			if (intdimOrNoteSetHeight < trialImageHeight / 20)
			{
				DimSet.dimSetLocations = cv::RotatedRect(p2ddimOrNoteSetCenter, cv::Size2f(((float)intdimOrNoteSetWidth), (float)intdimOrNoteSetHeight), (float)dblCorrectionAngleInDeg);
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

		rotationMatrix = cv::getRotationMatrix2D(p2ddimOrNoteSetCenter, dblCorrectionAngleInDeg, 1.0);         // get the rotation matrix for our calculated correction angle

		cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());            // rotate the entire image
																										// crop out the actual dimOrNoteSet portion of the rotated image
		cv::getRectSubPix(imgRotated, DimSet.dimSetLocations.size, DimSet.dimSetLocations.center, imgCropped);

		DimSet.imageDimSet = imgCropped;            // copy the cropped dimOrNoteSet image into the applicable member variable of the possible dimOrNoteSet
	
		return(DimSet);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//                                        Detect Characs                                         //
	///////////////////////////////////////////////////////////////////////////////////////////////////

	// global variables ///////////////////////////////////////////////////////////////////////////////

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
	std::vector<DimSet> detectChars(std::vector<DimSet> &vectorOfDimSets) {
		// this is only for showing steps
		std::vector<std::vector<cv::Point> > contours;
		cv::RNG rng;

		if (vectorOfDimSets.empty()) {               // if vector of possible dimOrNoteSets is empty
			return(vectorOfDimSets);                 // return
		}
		// at this point we can be sure vector of possible dimOrNoteSets has at least one dimOrNoteSet

		for (auto &DimSet : vectorOfDimSets) {            // for each possible dimOrNoteSet, this is a big for loop that takes up most of the function

			preprocess(DimSet.imageDimSet, DimSet.imageDimSetGSBW, DimSet.imageDimSetThreshold);        // preprocess to get grayscale and threshold images
																													// upscale size by 60% for better viewing and character recognition
			cv::resize(DimSet.imageDimSetThreshold, DimSet.imageDimSetThreshold, cv::Size(), 5, 5);

			// threshold again to eliminate any gray areas
			cv::threshold(DimSet.imageDimSetThreshold, DimSet.imageDimSetThreshold, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);

			// find all possible chars in the dimOrNoteSet,
			// this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
			std::vector<Character> vectorOfCharactersIndimOrNoteSet = findChars(DimSet.imageDimSetGSBW, DimSet.imageDimSetThreshold);
			contours.clear();

			for (auto &Character : vectorOfCharactersIndimOrNoteSet) {
				contours.push_back(Character.contour);
			}

			// given a vector of all possible chars, find groups of matching chars within the dimOrNoteSet
			std::vector<std::vector<Character> > vectorOfVectorsOfMatchingCharsIndimOrNoteSet = findVectorOfVectorsOfMatchingChars(vectorOfCharactersIndimOrNoteSet);

			//	vectorOfCharactersIndimOrNoteSet<<Character>>
			contours.clear();

			for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsIndimOrNoteSet) {
				for (auto &matchingChar : vectorOfMatchingChars) {
					contours.push_back(matchingChar.contour);
				}
			}

			if (vectorOfVectorsOfMatchingCharsIndimOrNoteSet.size() == 0) {                // if no groups of matching chars were found in the dimOrNoteSet

			//	std::cout << "chars found in dimOrNoteSet number " << intdimOrNoteSetCounter << " = (none), click on any image and press a key to continue . . ." << std::endl;
				intdimOrNoteSetCounter++;
				dummyCenter.x = 1;
				dummyCenter.y = 1;
				float 	dummyAngle = 0;

				DimSet.dimSetLocations = cv::RotatedRect(dummyCenter, cv::Size2f(1, 1), dummyAngle);
				// set dimOrNoteSet string member variable to empty string
				continue;                               // go back to top of for loop
			}

			for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsIndimOrNoteSet) {                                         // for each vector of matching chars in the current dimOrNoteSet
				std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), Character::sortCharsLeftToRight);      // sort the chars left to right
				vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                     // and eliminate any overlapping chars
			}

			for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsIndimOrNoteSet) {
				contours.clear();
				for (auto &matchingChar : vectorOfMatchingChars) {
					contours.push_back(matchingChar.contour);
				}
			}
			// within each possible dimOrNoteSet, suppose the longest vector of potential matching chars is the actual vector of chars
			unsigned int intLenOfLongestVectorOfChars = 0;
			unsigned int intIndexOfLongestVectorOfChars = 0;
			// loop through all the vectors of matching chars, get the index of the one with the most chars
			for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsIndimOrNoteSet.size(); i++) {
				if (vectorOfVectorsOfMatchingCharsIndimOrNoteSet[i].size() > intLenOfLongestVectorOfChars) {
					intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsIndimOrNoteSet[i].size();
					intIndexOfLongestVectorOfChars = i;
				}
			}
			// suppose that the longest vector of matching chars within the dimOrNoteSet is the actual vector of chars
			std::vector<Character> longestVectorOfMatchingCharsIndimOrNoteSet = vectorOfVectorsOfMatchingCharsIndimOrNoteSet[intIndexOfLongestVectorOfChars];

			contours.clear();

			for (auto &matchingChar : longestVectorOfMatchingCharsIndimOrNoteSet) {
				contours.push_back(matchingChar.contour);
			}
			// perform char recognition on the longest vector of matching chars in the dimOrNoteSet
		//	std::string vectorStrings = recognizeCharsInDimSet(DimSet.imageDimSetThreshold, longestVectorOfMatchingCharsIndimOrNoteSet);
			DimSet.dimNoteStrings = recognizeCharsInDimSet(DimSet.imageDimSetThreshold, longestVectorOfMatchingCharsIndimOrNoteSet);

		/*	DimSet.dimNoteStrings = vectorStrings[0].dimNoteStrings;
			DimSet.accuracyPercentage = vectorStrings[0].accuracyPercentage;
			DimSet.confidencePercentage = vectorStrings[0].confidencePercentage;*/

			//std::cout << "chars found in dimOrNoteSet number " << intdimOrNoteSetCounter << " = " << DimSet.dimNoteStrings << ", click on any image and press a key to continue . . ." << std::endl;
			intdimOrNoteSetCounter++;
		}   // end for each possible dimOrNoteSet big for loop that takes up most of the function

		//std::cout << std::endl << "char detection complete, click on any image and press a key to continue . . ." << std::endl;
		return(vectorOfDimSets);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<Character> findChars(cv::Mat &imageDimSetGSBW, cv::Mat &imageDimSetThreshold) {
		std::vector<Character> vectorOfCharacters;                            // this will be the return value

		cv::Mat imageDimSetThresholdCopy;

		std::vector<std::vector<cv::Point> > contours;

		imageDimSetThresholdCopy = imageDimSetThreshold.clone();				// make a copy of the thresh image, this in necessary b/c findContours modifies the image

		cv::findContours(imageDimSetThresholdCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);        // find all contours in dimOrNoteSet

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
																		  // if current possible vector of matching chars is not long enough to constitute a possible dimOrNoteSet
			if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) {
				continue;                       // jump back to the top of the for loop and try again with next char, note that it's not necessary
												// to save the vector in any way since it did not have enough chars to be a possible dimOrNoteSet
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
	// this is where we apply the actual char recognition
	//std::vector<dimNoteDimRecog::DimSet> recognizeCharsInDimSet(cv::Mat &imageDimSetThreshold, std::vector<Character> &vectorOfMatchingChars) {
	string recognizeCharsInDimSet(cv::Mat &imageDimSetThreshold, std::vector<Character> &vectorOfMatchingChars) {
		std::string dimNoteStrings;               // this will be the return value, the chars in the lic dimOrNoteSet
		std::vector<dimNoteDimRecog::DimSet> vectorStrings;
		double confidence = 0;
		double accuracyLevel = 0;

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
		{           // for each char in dimOrNoteSet
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
			double totdistance = 0;
			double accurateDistance = 0;

			int k = kNearest->getDefaultK();

			cv::Mat neighborResponses(1, k, CV_32FC1);

			kNearest->findNearest(matROIFlattenedFloat, k, matCurrentChar, neighborResponses, dist);     // finally we can call find_nearest !!!

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

			confidence = 0;
		    accuracyLevel = 0;

			confidence = (accurateDistance / totdistance) * 100;
			accuracyLevel = (accuracyLevel / k) * 100;

			float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       // convert current char from Mat to float

			dimNoteStrings = dimNoteStrings + char(int(fltCurrentChar));        // append current char to full string
		}	
	/*	vectorStrings[0].dimNoteStrings=dimNoteStrings;
		vectorStrings[0].accuracyPercentage = accuracyLevel;
		vectorStrings[0].confidencePercentage = confidence;*/
		return(dimNoteStrings);               // return result
	}
	std::vector<DimSet> dimensionNoteRecognition(cv::Mat &temp)
	{
		cv::Mat tempresize;

		std::vector<DimSet> vectorOfDimSets;
		inputImage = temp;
		tempresize = temp.clone();

		if (inputImage.empty()) {                             // if unable to open image
			std::cout << "error: image not read from file\n\n";     // show error message on command line
			_getch();                                               // may have to modify this line if not using Windows                                           
		}

		vectorOfDimSets = detectDimSet(inputImage);          // detect dimOrNoteSets

		vectorOfDimSets = detectChars(vectorOfDimSets);                               // detect chars in dimOrNoteSets

		if (vectorOfDimSets.empty())
		{                                               // if no dimOrNoteSets were found
			//std::cout << std::endl << "no dimOrNoteSets were detected" << std::endl;       // inform user no dimOrNoteSets were found
		}
		else {                                                                         // if we get in here vector of possible dimOrNoteSets has at leat one dimOrNoteSet
																					  // sort the vector of possible dimOrNoteSets in DESCENDING order (most number of chars to least number of chars)
			std::sort(vectorOfDimSets.begin(), vectorOfDimSets.end(), DimSet::sortDescendingByNumberOfChars);
			// suppose the dimOrNoteSet with the most recognized chars (the first dimOrNoteSet in sorted by string length descending order) is the actual dimOrNoteSet
			DimSet dimensionSet = vectorOfDimSets.front();

			for (size_t i = 0; i < vectorOfDimSets.size(); i++)
			{
				DimSet tempdimensionSet = vectorOfDimSets.at(i);
				drawRectAroundDimSet(inputImage, tempdimensionSet);
			}

			cv::getRectSubPix(temp, dimensionSet.dimSetLocations.size, dimensionSet.dimSetLocations.center, tempresize);

			if (dimensionSet.dimNoteStrings.length() == 0) {                                                      // if no chars were found in the dimOrNoteSet
				//std::cout << std::endl << "no characters were detected" << std::endl << std::endl;      // show message                                                                            
			}
			drawRectAroundDimSet(inputImage, dimensionSet);                // draw rectangle around dimOrNoteSet and make rectangle white
			string filename = "Image_";
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
			cv::imwrite(filename, inputImage);                  // write iteration image out to file
		}
		// hold windows open until user presses a key
		return vectorOfDimSets;
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
		DimSet tempDimSet;// return value

		std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
		std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly
		cv::Mat matGrayscale = src;          //
		cv::Mat matBlurred;             // declare more image variables
		cv::Mat matThresh;              //
		cv::Mat matThreshCopy=src;          //

	/*	// blur
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
		*/

		std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
		std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)
																		
		cv::imshow("matThreshCopy", matThreshCopy);
		cv::waitKey(0);

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

		for (int i = 0; i < validContoursWithData.size(); i++)
		{            // for each contour
			cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect

			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

			cv::Mat matROIFloat;
			matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

			cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

			cv::Mat matCurrentChar(0, 0, CV_32F);

			kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

			float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

			strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
		}
		tempDimSet.dimNoteStrings = strFinalString;
		return tempDimSet;
	}

	std::vector<DimSet> dimNoteDimRecognition() {
		char inputFilename[255];
		int horVerMode;
		int knn;
		//std::cout << "\n Enter k for kNN\n " << endl;
		//cin >> knn;
		
		MessageBox(NULL, L"Please open in the following dialog box, the drawing for which\n dimensions, tolerancing and notes are to be recognized \n\n Supported formats:*.bmp,*.dib,*.jpeg,*.jpg,*.jpe,*.jp2,*.png,\n*.webp,*.pbm,*.pgm,*.ppm,*.sr,*.ras,*.tiff, *.tif", L"Open Drawing", MB_OK);
		
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
							WideCharToMultiByte(CP_ACP,0, pszFilePath, -1, inputFilename, sizeof(inputFilename), NULL, NULL);

							CoTaskMemFree(pszFilePath);
						}
						pItem->Release();
					}
				}
				pFileOpen->Release();
			}
			CoUninitialize();
		}		

		std::cout << "\n Enter horVerMode:\n0 for only horizontal\n1 for only horizontal and vertical dimensions\n2 for all orientations based search \n" << endl;
		cin >> horVerMode;
		std::cout << "\n Enter k for kNN based search \n" << endl;
		cin >> knn;

		std::vector<DimSet> finalOutput;
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

		Mat gray = src;

		if (src.channels() == 3)
		{
			cvtColor(gray, gray, CV_BGR2GRAY);
		}
		else
		{
			gray = src;
		}
	
		// Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
		Mat bw;
		adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	
		// Create the images that will use to extract the horizontal and vertical lines
		Mat horizontal = bw.clone();
		Mat vertical = bw.clone();
		Mat horVer = bw.clone();
		// Specify size on horizontal axis
		int horizontalsize = horizontal.cols / 100;
		// Create structure element for extracting horizontal lines through morphology operations
		Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
		// Apply morphology operations
		erode(horizontal, horizontal, horizontalStructure, Point(-1, -1), 10);
		dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1), 10);

		// Specify size on vertical axis
		int verticalsize = vertical.rows / 100;
		Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
		erode(vertical, vertical, verticalStructure, Point(-1, -1), 10);
		dilate(vertical, vertical, verticalStructure, Point(-1, -1), 10);
	
		horVer = vertical + horizontal;
		subtract(bw, horVer, horVer);
	
		bitwise_not(horVer, horVer);
	
		vertical = horVer;
		bitwise_not(vertical, vertical);
	
		Mat edges;
		adaptiveThreshold(vertical, edges, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);

		Mat kernel = Mat::ones(2, 2, CV_8UC1);
		dilate(edges, edges, kernel);

		Mat smooth;
		vertical.copyTo(smooth);
		blur(smooth, smooth, Size(2, 2));
		smooth.copyTo(vertical, edges);
		adaptiveThreshold(vertical, vertical, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
		bitwise_not(vertical, vertical);
		vertical = maximizeContrast(vertical);
		cvtColor(vertical, vertical, CV_GRAY2BGR);

		temp1=maximizeContrast(vertical);
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

		if (horizontalVerticalMode >= 0)
		{
			iterIndex = 1;
			trialImageWidth = temp.size().width;
			trialImageHeight = temp.size().height;
			output.vectorOfDimSets1 = dimensionNoteRecognition(temp);
			finalOutput.insert(finalOutput.begin(), output.vectorOfDimSets1.begin(), output.vectorOfDimSets1.end());
			finalOutput.push_back(dummyDimSet);

		}

		dummyLocations[0] = finalOutput.size();

		temprot1 = inputImage;
		transpose(temprot1, temprot);
		flip(temprot, temprot, 1);

		for (size_t i = 0; i < dummyLocations[0]; i++)
		{
			cv::Rect boundingRect = finalOutput[i].dimSetLocations.boundingRect();
			for (size_t j = 0; j < dummyLocations[0]; j++)
			{
				cv::Rect boundingRect2 = finalOutput[j].dimSetLocations.boundingRect();
				bool tlContainer = boundingRect.contains(boundingRect2.tl());
				bool brContainer = boundingRect.contains(boundingRect2.br());

				if (tlContainer*brContainer)
				{
					finalOutput[j].dimSetLocations = dummyDimSet.dimSetLocations;
					finalOutput[j].dimNoteStrings = "dummy";
				}
			}
		}

		inputImage = temp1;

		if (horizontalVerticalMode > 1)
		{
			iterIndex = 2;
			trialImageWidth = inputImage.size().width;
			trialImageHeight = inputImage.size().height;
			output.vectorOfDimSets3 = dimensionNoteRecognition(inputImage);
			finalOutput.insert(finalOutput.end(), output.vectorOfDimSets3.begin(), output.vectorOfDimSets3.end());
			finalOutput.push_back(dummyDimSet);
		}

		dummyLocations[1] = finalOutput.size();

		iter4 = inputImage;

		for (size_t i = dummyLocations[0]; i < dummyLocations[1]; i++)
		{
			cv::Rect boundingRect = finalOutput[i].dimSetLocations.boundingRect();
			for (size_t j = 0; j < dummyLocations[0]; j++)
			{
				cv::Rect boundingRect2 = finalOutput[j].dimSetLocations.boundingRect();
				bool tlContainer = boundingRect.contains(boundingRect2.tl());
				bool brContainer = boundingRect.contains(boundingRect2.br());

				if (tlContainer*brContainer)
				{
					finalOutput[j].dimSetLocations = dummyDimSet.dimSetLocations;
					finalOutput[j].dimNoteStrings = "dummy";
				}
			}
		}

		if (horizontalVerticalMode > 0)
		{
			iterIndex = 3;
			trialImageWidth = temprot.size().width;
			trialImageHeight = temprot.size().height;
			output.vectorOfDimSets2 = dimensionNoteRecognition(temprot);
			finalOutput.insert(finalOutput.end(), output.vectorOfDimSets2.begin(), output.vectorOfDimSets2.end());
			finalOutput.push_back(dummyDimSet);
		}
		dummyLocations[2] = finalOutput.size();

		for (size_t i = dummyLocations[1]; i < dummyLocations[2]; i++)
		{
			cv::Rect boundingRect = finalOutput[i].dimSetLocations.boundingRect();
			for (size_t j = 0; j < dummyLocations[0]; j++)
			{
				cv::Rect boundingRect2 = finalOutput[j].dimSetLocations.boundingRect();
				bool tlContainer = boundingRect.contains(boundingRect2.tl());
				bool brContainer = boundingRect.contains(boundingRect2.br());

				if (tlContainer*brContainer)
				{
					finalOutput[j].dimSetLocations = dummyDimSet.dimSetLocations;
					finalOutput[j].dimNoteStrings = "dummy";
				}
			}
		}

		transpose(iter4, iter4);
		flip(iter4, iter4, 1);

		if (horizontalVerticalMode > 1)
		{
			iterIndex = 4;
			trialImageWidth = iter4.size().width;
			trialImageHeight = iter4.size().height;
			output.vectorOfDimSets4 = dimensionNoteRecognition(iter4);
			finalOutput.insert(finalOutput.end(), output.vectorOfDimSets4.begin(), output.vectorOfDimSets4.end());
		}
		for (size_t i = dummyLocations[2]; i < finalOutput.size(); i++)
		{
			cv::Rect boundingRect = finalOutput[i].dimSetLocations.boundingRect();
			for (size_t j = 0; j < dummyLocations[0]; j++)
			{
				cv::Rect boundingRect2 = finalOutput[j].dimSetLocations.boundingRect();
				bool tlContainer = boundingRect.contains(boundingRect2.tl());
				bool brContainer = boundingRect.contains(boundingRect2.br());

				if (tlContainer*brContainer)
				{
					finalOutput[j].dimSetLocations = dummyDimSet.dimSetLocations;
					finalOutput[j].dimNoteStrings = "dummy";
				}
			}
		}

		std::vector<cv::RotatedRect> rotRects;

		for (size_t i = dummyLocations[1]; i < finalOutput.size(); i++)
		{
			cv::Point2f updatedCenter;

			double swap = finalOutput[i].dimSetLocations.center.x;
			updatedCenter.x = finalOutput[i].dimSetLocations.center.y;
			updatedCenter.y = trialImage.size().height - swap;

			cv::RotatedRect tempRect = cv::RotatedRect(updatedCenter, cv::Size2f(finalOutput[i].dimSetLocations.size.height, finalOutput[i].dimSetLocations.size.width), (finalOutput[i].dimSetLocations.angle - 180));
			finalOutput[i].dimSetLocations = tempRect;

		}

		for (size_t i = 0; i < finalOutput.size(); i++)
		{
			drawRotatedRect(trialImage, finalOutput[i].dimSetLocations, SCALAR_WHITE, 1);
		}

		std::string opfilename = inputFilename;
		opfilename[(opfilename.size() - 4)] = '_';
		opfilename.append("_Output.jpg");
		cv::imwrite(opfilename, trialImage);

		cv::Point2f completeWhiteImageCenter;
		completeWhiteImageCenter.x = trialImage.size().width / 2;
		completeWhiteImageCenter.y = trialImage.size().height / 2;

		cv::RotatedRect completeWhiteImage = cv::RotatedRect(completeWhiteImageCenter, cv::Size2f(trialImage.size().width, trialImage.size().height), 0);

		drawRotatedRect(trialImage, completeWhiteImage, SCALAR_WHITE, 0);

		for (size_t i = 0; i < finalOutput.size(); i++)
		{
			drawRectAroundDimSet(trialImage, finalOutput[i]);
		}
		cv::imwrite("OutputDimensions.jpg", trialImage);

		std::vector<double> finalOutputBoundingRectHeights;
		double minFinalOutputBoundingRectHeight;

		cv::Mat BWtrialImage;

		for (size_t i = 0; i < finalOutput.size(); i++)
		{
			drawRectAroundDimSet(BWtrialImage, finalOutput[i]);
			minFinalOutputBoundingRectHeight = finalOutput[i].dimSetLocations.size.height;
			finalOutputBoundingRectHeights.push_back(minFinalOutputBoundingRectHeight);
		}

		minFinalOutputBoundingRectHeight = *(std::min(finalOutputBoundingRectHeights.begin(), finalOutputBoundingRectHeights.end()));

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

		dummyContourPoints.push_back(dummyCounterPt1);
		dummyContourPoints.push_back(dummyCounterPt2);

		for (unsigned int i = 0; i < contours.size(); i++)
		{
			if (hierarchy[i][3] == -1)
			{
				int noOfLines;
				finalContourSize++;
				cv::RotatedRect boundRotRect = minAreaRect(contours[i]);
				finContourRotRecs.push_back(boundRotRect);				
			}
			else
			{
				contours[i] = dummyContourPoints;
			}
		}
		for (unsigned int i = 0; i < finContourRotRecs.size(); i++)
		{
			drawRotatedRect(imgWhite2, finContourRotRecs[i], SCALAR_GREEN, 0);
		}
		double alpha = 0.5;
		double beta = (1.0 - alpha);
		addWeighted(imgWhite2, alpha,srctemp, beta, 0.0, srctemp);
		srctemp=maximizeContrast(srctemp);
		cv::imwrite("PostProcessed.jpg", srctemp);
		
		Mat frame = srctemp;
		namedWindow("Trial_ROI_Mouse", WINDOW_KEEPRATIO);
		setMouseCallback("Trial_ROI_Mouse", onMouse, 0);
		MessageBox(NULL, L"\n i or I key should be pressed to drag and select unrecognized dimension \n, x or X key is to stop selection and enable character recognition.  \n ESC key is exit.\n", L"Important message !", MB_OK);
			
		for (;;)
		{
			frame.copyTo(image);

			if (insert && selection.width > 0 && selection.height > 0)
			{
				rectangle(image, Point(selection.x - 1, selection.y - 1), Point(selection.x + selection.width + 1, selection.y + selection.height + 1), CV_RGB(255, 0, 0));
			}

			imshow("Trial_ROI_Mouse", image);

			char k = waitKey(0);
			string filename = "Trial_ROI_Mouse";

			if (k == 27)//Escape key breaks the loop and ends
			{
				cv::destroyAllWindows;
				break;
			}
			if (k == 'x' || k == 'X')// x key shows selected images
			{
				cout << "\n\n Size of setOfnewDimSets:\n " << setOfnewDimSets.size() << endl;
				for (int i = 0; i < setOfnewDimSets.size(); i++)
				{
					if (setOfnewDimSets[i].height != 0)
					{
						cv::Mat newDimSetImgRotCrop = RotToHor(setOfnewDimSets[i].newDimSetImg);
				//		findContours(newDimSetImgRotCrop, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
				//		drawContours(newDimSetImgRotCrop, contours, -1, SCALAR_BLACK, 3, 8, hierarchy, 0);											
						imshow("drawContours", newDimSetImgRotCrop);
						DimSet tempDimSet = newDimNoteDimRecognition(newDimSetImgRotCrop);
	
						finalOutput.push_back(tempDimSet);
						cout << "\n\nThe new strings:\n" << tempDimSet.dimNoteStrings << endl;
						waitKey(0);
						cv::destroyAllWindows;
					}
				}
			}
			else if (k == 'i' || k == 'I')//I key enables selection
				insert = !insert;
		}
		cv::destroyAllWindows;
		return finalOutput;

	}
}