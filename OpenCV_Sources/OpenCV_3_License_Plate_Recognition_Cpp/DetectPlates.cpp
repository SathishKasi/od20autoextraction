// DetectPlates.cpp

#include "DetectPlates.h"
#include "Main.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <math.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

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

vector<Point2d> eigen_vecs(2);
vector<double> eigen_val(2);
Point pos;

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
	// Draw the principal components
	//circle(img, pos, 3, CV_RGB(255, 0, 255), 2);
	//line(img, pos, pos + 0.02 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]), CV_RGB(255, 0, 0),4);
	//line(img, pos, pos + 0.02 * Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]), CV_RGB(0, 255, 0),4);

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
	return daArray[iMaxRepeat];
}
using namespace cv;
using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PossiblePlate> detectPlatesInScene(cv::Mat &imgOriginalScene) {
    std::vector<PossiblePlate> vectorOfPossiblePlates;			// this will be the return value

    cv::Mat imgGrayscaleScene;
    cv::Mat imgThreshScene;
    cv::Mat imgContours(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);

    cv::RNG rng;

    cv::destroyAllWindows();

#ifdef SHOW_STEPS
   // //cv::imshow("0", imgOriginalScene);
#endif	// SHOW_STEPS

    preprocess(imgOriginalScene, imgGrayscaleScene, imgThreshScene);        // preprocess to get grayscale and threshold images

#ifdef SHOW_STEPS
    ////cv::imshow("imgGrayscaleScene", imgGrayscaleScene);
 //   //cv::imshow("imgThreshScene", imgThreshScene);
//	cv::imwrite("imgThreshScene.jpg", imgThreshScene);
#endif	// SHOW_STEPS

    // find all possible chars in the scene,
    // this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    std::vector<PossibleChar> vectorOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene);

	cout << "\n\n vectorOfPossibleCharsInScene.size()\n";
	cout << vectorOfPossibleCharsInScene.size();

	//for (size_t i = 0; i < vectorOfPossibleCharsInScene.size(); i++)
	//{
	//double AngleDeg = getOrientation(vectorOfPossibleCharsInScene[i].PossibleChar::contour, imgContours);
	cout << "\n\n X values : \n" << endl;
	cout << vectorOfPossibleCharsInScene[0].PossibleChar::contour[0].Point::x;
	//}
	
    std::cout << "step 2 - vectorOfPossibleCharsInScene.Count = " << vectorOfPossibleCharsInScene.size() << std::endl;        // 131 with MCLRNF1 image

    imgContours = cv::Mat(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);
    std::vector<std::vector<cv::Point> > contours;

    for (auto &possibleChar : vectorOfPossibleCharsInScene) {
        contours.push_back(possibleChar.contour);
    }
    cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);
   // //cv::imshow("imgContours", imgContours);
	cv::imwrite("imgContours.jpg", imgContours);


    // given a vector of all possible chars, find groups of matching chars
    // in the next steps each group of matching chars will attempt to be recognized as a plate
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInScene);

#ifdef SHOW_STEPS
    std::cout << "step 3 - vectorOfVectorsOfMatchingCharsInScene.size() = " << vectorOfVectorsOfMatchingCharsInScene.size() << std::endl;        // 13 with MCLRNF1 image

    imgContours = cv::Mat(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);

    for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {
        int intRandomBlue = rng.uniform(0, 256);
        int intRandomGreen = rng.uniform(0, 256);
        int intRandomRed = rng.uniform(0, 256);

        std::vector<std::vector<cv::Point> > contours;

        for (auto &matchingChar : vectorOfMatchingChars) {
            contours.push_back(matchingChar.contour);
        }
        cv::drawContours(imgContours, contours, -1, cv::Scalar(255,255,255));
    }
  //  //cv::imshow("imgContours", imgContours);
#endif	// SHOW_STEPS

    for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {                     // for each group of matching chars
        PossiblePlate possiblePlate = extractPlate(imgOriginalScene, vectorOfMatchingChars);        // attempt to extract plate

        if (possiblePlate.imgPlate.empty() == false) {                                              // if plate was found
            vectorOfPossiblePlates.push_back(possiblePlate);                                        // add to vector of possible plates
        }
    }

    std::cout << std::endl << vectorOfPossiblePlates.size() << " Possible plates found" << std::endl;       // 13 with MCLRNF1 image

#ifdef SHOW_STEPS
    std::cout << std::endl;
 //   //cv::imshow("Possible plates found", imgContours);
	Mat imgContourstemp;
	cvtColor(imgContours, imgContourstemp, CV_BGR2GRAY);
	adaptiveThreshold(imgContourstemp, imgContourstemp, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	bitwise_not(imgContourstemp, imgContourstemp);
	cv::imwrite("Possible plates found.jpg", imgContourstemp);

    for (unsigned int i = 0; i < vectorOfPossiblePlates.size(); i++) {
        cv::Point2f p2fRectPoints[4];

        vectorOfPossiblePlates[i].rrLocationOfPlateInScene.points(p2fRectPoints);

        for (int j = 0; j < 4; j++) {
            cv::line(imgContours, p2fRectPoints[j], p2fRectPoints[(j + 1) % 4], SCALAR_RED, 2);
        }
       // //cv::imshow("Red Rect vectorOfPossiblePlates rrLocationOfPlateInScene points", imgContours);
		////cv::imshow("Red Rect vectorOfPossiblePlates rrLocationOfPlateInScene points.jpg", imgContours);

        std::cout << "possible plate " << i << ", click on any image and press a key to continue . . ." << std::endl;

     //   //cv::imshow("vectorOfPossiblePlates imgPlate", vectorOfPossiblePlates[i].imgPlate);
	//	cv::imwrite("vectorOfPossiblePlates imgPlate.jpg", vectorOfPossiblePlates[i].imgPlate);

        cv::waitKey(0);
    }
    std::cout << std::endl << "plate detection complete, click on any image and press a key to begin char recognition . . ." << std::endl << std::endl;
    cv::waitKey(0);
#endif	// SHOW_STEPS

    return vectorOfPossiblePlates;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PossibleChar> findPossibleCharsInScene(cv::Mat &imgThresh) {
    std::vector<PossibleChar> vectorOfPossibleChars;            // this will be the return value

    cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);
    int intCountOfPossibleChars = 0;

    cv::Mat imgThreshCopy = imgThresh.clone();

    std::vector<std::vector<cv::Point> > contours;

    cv::findContours(imgThreshCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);        // find all contours

    for (unsigned int i = 0; i < contours.size(); i++) {                // for each contour
#ifdef SHOW_STEPS
        cv::drawContours(imgContours, contours, i, SCALAR_WHITE);
#endif	// SHOW_STEPS
        PossibleChar possibleChar(contours[i]);

        if (checkIfPossibleChar(possibleChar)) {                // if contour is a possible char, note this does not compare to other chars (yet) . . .
            intCountOfPossibleChars++;                          // increment count of possible chars
            vectorOfPossibleChars.push_back(possibleChar);      // and add to vector of possible chars
        }
    }

#ifdef SHOW_STEPS
    std::cout << std::endl << "contours.size() = " << contours.size() << std::endl;                         // 2362 with MCLRNF1 image
    std::cout << "step 2 - intCountOfValidPossibleChars = " << intCountOfPossibleChars << std::endl;        // 131 with MCLRNF1 image
    ////cv::imshow("2a", imgContours);
#endif	// SHOW_STEPS

    return(vectorOfPossibleChars);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
PossiblePlate extractPlate(cv::Mat &imgOriginal, std::vector<PossibleChar> &vectorOfMatchingChars) {
    PossiblePlate possiblePlate;            // this will be the return value

                                            // sort chars from left to right based on x position
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    // calculate the center point of the plate
    double dblPlateCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
    double dblPlateCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;
    cv::Point2d p2dPlateCenter(dblPlateCenterX, dblPlateCenterY);

    // calculate plate width and height
    int intPlateWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * PLATE_WIDTH_PADDING_FACTOR);

    double intTotalOfCharHeights = 0;

    for (auto &matchingChar : vectorOfMatchingChars) {
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.boundingRect.height;
    }

    double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();

    int intPlateHeight = (int)(dblAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR);

    // calculate correction angle of plate region
    double dblOpposite = vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY - vectorOfMatchingChars[0].intCenterY;
    double dblHypotenuse = distanceBetweenChars(vectorOfMatchingChars[0], vectorOfMatchingChars[vectorOfMatchingChars.size() - 1]);
    double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
    double dblCorrectionAngleInDeg = (-dblCorrectionAngleInRad * (180.0 / CV_PI))-90;

	/*
	if ((abs(dblCorrectionAngleInDeg)>5)&& (abs(dblCorrectionAngleInDeg)<175))
	{
		std::cout << std::endl << "CorrectionAngleInDeg = " << dblCorrectionAngleInDeg << std::endl;
	}
	*/
    // assign rotated rect member variable of possible plate
    possiblePlate.rrLocationOfPlateInScene = cv::RotatedRect(p2dPlateCenter, cv::Size2f((float)intPlateWidth, (float)intPlateHeight), (float)dblCorrectionAngleInDeg);

    cv::Mat rotationMatrix;             // final steps are to perform the actual rotation
    cv::Mat imgRotated;
    cv::Mat imgCropped;

    rotationMatrix = cv::getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);         // get the rotation matrix for our calculated correction angle

    cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());            // rotate the entire image

                                                                                            // crop out the actual plate portion of the rotated image
    cv::getRectSubPix(imgRotated, possiblePlate.rrLocationOfPlateInScene.size, possiblePlate.rrLocationOfPlateInScene.center, imgCropped);

    possiblePlate.imgPlate = imgCropped;            // copy the cropped plate image into the applicable member variable of the possible plate

    return(possiblePlate);
}

