// Main.cpp

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



///////////////////////////////////////////////////////////////////////////////////////////////////

void LicplatRecog(cv::Mat temp)
{
	cv::Mat tempresize;
	cv::Mat imgOriginalScene;

	////cv::imshow("imgOriginalScene", temp);
	cv::resize(temp, imgOriginalScene, cv::Size(), 3, 3);

	if (imgOriginalScene.empty()) {                             // if unable to open image
		std::cout << "error: image not read from file\n\n";     // show error message on command line
		_getch();                                               // may have to modify this line if not using Windows                                           
	}

	std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);          // detect plates

	vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                               // detect chars in plates

	if (vectorOfPossiblePlates.empty())
	{                                               // if no plates were found
		std::cout << std::endl << "no license plates were detected" << std::endl;       // inform user no plates were found
	}
	else {                                                                            // else
																					  // if we get in here vector of possible plates has at leat one plate

																					  // sort the vector of possible plates in DESCENDING order (most number of chars to least number of chars)
		std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);

		// suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
		PossiblePlate licPlate = vectorOfPossiblePlates.front();

		for (size_t i = 0; i < vectorOfPossiblePlates.size(); i++)
		{
			PossiblePlate templicPlate = vectorOfPossiblePlates.at(i);
			drawRedRectangleAroundPlate(imgOriginalScene, templicPlate);
		}

		cv::getRectSubPix(temp, licPlate.rrLocationOfPlateInScene.size, licPlate.rrLocationOfPlateInScene.center, tempresize);

		/*
		////cv::imshow("img_In", licPlate.imgPlate); // show crop of plate and threshold of plate
		////cv::imshow("imgThresh", licPlate.imgThresh);

		PossiblePlate licPlate2 = vectorOfPossiblePlates.at(1);
		////cv::imshow("licPlate2", licPlate2.imgPlate);
		drawRedRectangleAroundPlate(imgOriginalScene, licPlate2);

		PossiblePlate licPlate3 = vectorOfPossiblePlates.at(2);
		////cv::imshow("licPlate3", licPlate3.imgPlate);
		drawRedRectangleAroundPlate(imgOriginalScene, licPlate3);

		PossiblePlate licPlate4 = vectorOfPossiblePlates.at(3);
		////cv::imshow("licPlate4", licPlate4.imgPlate);
		drawRedRectangleAroundPlate(imgOriginalScene, licPlate4);

		PossiblePlate licPlate5 = vectorOfPossiblePlates.at(4);
		////cv::imshow("licPlate5", licPlate5.imgPlate);
		drawRedRectangleAroundPlate(imgOriginalScene, licPlate5);

		PossiblePlate licPlate6 = vectorOfPossiblePlates.at(5);
		////cv::imshow("licPlate6", licPlate6.imgPlate);
		drawRedRectangleAroundPlate(imgOriginalScene, licPlate6);
		*/

		if (licPlate.strChars.length() == 0) {                                                      // if no chars were found in the plate
			std::cout << std::endl << "no characters were detected" << std::endl << std::endl;      // show message                                                                            // and exit program
		}

		drawRedRectangleAroundPlate(imgOriginalScene, licPlate);                // draw red rectangle around plate

		std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;     // write license plate text to std out
		std::cout << std::endl << "-----------------------------------------" << std::endl;

		writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);              // write license plate text on the image

		//cv::namedWindow("Display Image", CV_WINDOW_NORMAL);
		////cv::imshow("Display Image", imgOriginalScene);                       // re-show scene image

	//	cv::imwrite("imgOriginalScene.png", imgOriginalScene);                  // write image out to file
	}

	cv::waitKey(0);                 // hold windows open until user presses a key


}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
	cv::Point2f p2fRectPoints[4];

	licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);            // get 4 vertices of rotated rect

	for (int i = 0; i < 4; i++) {                                       // draw 4 red lines
		cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
	cv::Point ptCenterOfTextArea;                   // this will be the center of the area the text will be written to
	cv::Point ptLowerLeftTextOrigin;                // this will be the bottom left of the area that the text will be written to

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;                              // choose a plain jane font
	double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;            // base font scale on height of plate area
	int intFontThickness = (int)std::round(dblFontScale * 1.5);             // base font thickness on font scale
	int intBaseline = 0;

	cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);      // call getTextSize

	ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         // the horizontal location of the text area is the same as the plate

	if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      // if the license plate is in the upper 3/4 of the image
																							// write the chars in below the plate
		ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
	}
	else {                                                                                // else if the license plate is in the lower 1/4 of the image
																						  // write the chars in above the plate
		ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
	}

	ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));           // calculate the lower left origin of the text area
	ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));          // based on the text area center, width, and height

																							// write the text on the image
	cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
	cv::imwrite("imgOriginalScene.png", imgOriginalScene);
}
int main() {

    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();           // attempt KNN training

    if (blnKNNTrainingSuccessful == false) {                            // if KNN training was not successful
                                                                        // show error message
        std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
        return(0);                                                      // and exit program
    }

	cv::Mat temp;
	cv::Mat temptrans;
	cv::Mat temprot;
    cv::Mat imgOriginalScene;           // input image

    temp = cv::imread("resized_deskew1.jpg");         // open image
	
	cv::Point2f srcTri[3];
	cv::Point2f dstTri[3];

	srcTri[0] = cv::Point2f(0, 0);
	srcTri[1] = cv::Point2f(0, 1);
	srcTri[2] = cv::Point2f(1, 0);

	dstTri[0] = cv::Point2f((temp.rows*0.5), 0);
	dstTri[1] = cv::Point2f((temp.rows*0.5), 1);
	dstTri[2] = cv::Point2f(((temp.rows*0.5) + 1), 0);

	cv::Mat trans_mat = getAffineTransform(srcTri, dstTri);
	warpAffine(temp, temptrans, trans_mat, cv::Size((3 * temp.rows), (1.5*temp.cols)));

	cv::Point2f pt(temptrans.cols / 2.0F, temptrans.rows / 2.0F);
	cv::Mat M = cv::getRotationMatrix2D(pt, 90, 1.0);

	warpAffine(temptrans,temprot, M, cv::Size((1.5 * temptrans.rows), (1.5*temptrans.cols)));

	/*//cv::imshow("Translated Image", temptrans);
	cv::waitKey(0);
	//cv::imshow("Rotated Image", temprot);
	cv::waitKey(0);
	*/
	LicplatRecog(temp);
	cv::waitKey(0);
	cv::waitKey(0);
	cv::imshow("Input Image", temp);
	cv::waitKey(0);
	//LicplatRecog(temprot);
	//cv::waitKey(0);
	
}


