
#pragma once
// dimNoteDimRecog.h - Contains declaration of Function class  
#pragma once 

#ifdef dimNoteDimRecog_EXPORTS  
#define dimNoteDimRecog_API __declspec(dllexport)   
#else  
#define dimNoteDimRecog_API __declspec(dllimport)   
#endif

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           // may have to modify this line if not using Windows
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <math.h>

#include<opencv2/ml/ml.hpp>
#include <string>
#include<sstream>
#include <fstream>
#include <Tchar.h>
#include <io.h>
#include <fcntl.h>
#include <locale>
#include <codecvt>

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

using namespace cv;
using namespace std;

namespace dimNoteDimRecog
{
	// This class is exported from the dimNoteDimRecog.dll  
	// global constants ///////////////////////////////////////////////////////////////////////////////
	const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
	const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
	const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
	const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
	const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
	const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

	const double dimOrNoteSet_WIDTH_PADDING_FACTOR = 1.3;
	const double dimOrNoteSet_HEIGHT_PADDING_FACTOR = 1.5;
	const int MIN_PIXEL_WIDTH = 2;
	const int MIN_PIXEL_HEIGHT = 8;

	const double MIN_ASPECT_RATIO = 0.1;
	const double MAX_ASPECT_RATIO = 0.9;

	const int MIN_PIXEL_AREA = 100;

	// constants for comparing two chars
	const double MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.01;
	const double MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0;

	const double MAX_CHANGE_IN_AREA = 0.25;
	const double MAX_AREA = 2000;
	const double MAX_CHANGE_IN_WIDTH = 0.8;
	const double MAX_CHANGE_IN_HEIGHT = 0.2;

	// other constants
	const int MIN_NUMBER_OF_MATCHING_CHARS = 2;

	const int RESIZED_CHAR_IMAGE_WIDTH = 20;
	const int RESIZED_CHAR_IMAGE_HEIGHT = 30;

	const int MIN_CONTOUR_AREA = 150;
	// global variables ///////////////////////////////////////////////////////////////////////////////
	const cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
	const int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
	const int ADAPTIVE_THRESH_WEIGHT = 9;

	// external global variables //////////////////////////////////////////////////////////////////////
	extern const bool blnShowSteps;
	extern cv::Ptr<cv::ml::KNearest>  kNearest;

	///////////////////////////////////////////////////////////////////////////////////////////////////

	class DimSet {
	public:
		// member variables ///////////////////////////////////////////////////////////////////////////
		cv::Mat imageDimSet;
		cv::Mat imageDimSetGSBW;
		cv::Mat imageDimSetThreshold;
		cv::RotatedRect dimSetLocations;
		std::string dimNoteStrings;
		std::string accuracyPercentage;
		std::string confidencePercentage;

		///////////////////////////////////////////////////////////////////////////////////////////////
		static bool sortDescendingByNumberOfChars(const DimSet &ppLeft, const DimSet &ppRight) {
			return(ppLeft.dimNoteStrings.length() > ppRight.dimNoteStrings.length());
		}

	};
	///////////////////////////////////////////////////////////////////////////////////////////////////
	class opVector {
	public:
		std::vector<DimSet> vectorOfDimSets1;
		std::vector<DimSet> vectorOfDimSets2;
		std::vector<DimSet> vectorOfDimSets3;
		std::vector<DimSet> vectorOfDimSets4;
	};
	///////////////////////////////////////////////////////////////////////////////////////////////////
	class Character {
	public:
		// member variables ///////////////////////////////////////////////////////////////////////////
		std::vector<cv::Point> contour;

		cv::Rect boundingRect;

		int intCenterX;
		int intCenterY;

		double dblDiagonalSize;
		double dblAspectRatio;

		///////////////////////////////////////////////////////////////////////////////////////////////
		static bool sortCharsLeftToRight(const Character &pcLeft, const Character & pcRight) {
			return(pcLeft.intCenterX < pcRight.intCenterX);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		bool operator == (const Character& otherCharacter) const {
			if (this->contour == otherCharacter.contour) return true;
			else return false;
		}

		///////////////////////////////////////////////////////////////////////////////////////////////
		bool operator != (const Character& otherCharacter) const {
			if (this->contour != otherCharacter.contour) return true;
			else return false;
		}

		// function prototypes ////////////////////////////////////////////////////////////////////////
		Character(std::vector<cv::Point> _contour);

	};

	//class Functions
	//{
	// function prototypes ////////////////////////////////////////////////////////////////////////////
	std::vector<DimSet> dimNoteDimRecognition();

	void drawRectAroundDimSet(cv::Mat &inputImage, DimSet &dimensionSet);

	std::vector<DimSet> detectDimSet(cv::Mat &inputImage);

	std::vector<Character> findCharactersFromInput(cv::Mat &imageDimSetThreshold);

	DimSet extractDimSet(cv::Mat &imgOriginal, std::vector<Character> &vectorOfMatchingChars);

	bool loadKNNDataAndTrainKNN(void);

	std::vector<DimSet> detectChars(std::vector<DimSet> &vectorOfDimSets);

	std::vector<Character> findChars(cv::Mat &imageDimSetGSBW, cv::Mat &imageDimSetThreshold);

	bool checkIfCharacter(Character &Character);

	std::vector<std::vector<Character> > findVectorOfVectorsOfMatchingChars(const std::vector<Character> &vectorOfCharacters);

	std::vector<Character> findVectorOfMatchingChars(const Character &charac, const std::vector<Character> &vectorOfChars);

	double distanceBetweenChars(const Character &firstChar, const Character &secondChar);

	double angleBetweenChars(const Character &firstChar, const Character &secondChar);

	std::vector<Character> removeInnerOverlappingChars(std::vector<Character> &vectorOfMatchingChars);

	std::vector<std::string> recognizeCharsInDimSet(cv::Mat &imageDimSetThreshold, std::vector<Character> &vectorOfMatchingChars);
	//};
}
