// DetectChars.cpp

#include "DetectChars.h"
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


// global variables ///////////////////////////////////////////////////////////////////////////////
cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();

/*
void RotToHor(Mat bw)
{
	double EigVecX = 0;
	double EigVecY = 0;
	Mat dst, fdst;

	cvtColor(bw, bw, COLOR_BGR2GRAY);
	// Convert it to greyscale
	threshold(bw, bw, 150, 255, CV_THRESH_BINARY);
	////cv::imshow("Input Image", bw);
	//cv::waitKey(0);

	double AngleDeg = 0;

	/*
	Point2f srcTri[3];
	Point2f dstTri[3];

	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(0, 1);
	srcTri[2] = Point2f(1, 0);

	dstTri[0] = Point2f((bw.rows*0.25), 0);
	dstTri[1] = Point2f((bw.rows*0.25), 1);
	dstTri[2] = Point2f(((bw.rows*0.25) + 1), 0);

	Mat trans_mat = getAffineTransform(srcTri, dstTri);
	warpAffine(bw, bw, trans_mat, Size((1.5 * bw.rows), (1.5 * bw.cols)));
	*/
/*
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

	// Find all objects of interest
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(bw, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// For each object
	for (size_t i = 0; i < contours.size(); ++i)
	{
		// Calculate its area
		area = contourArea(contours[i]);

		// Ignore if too small or too large
		if (area < 1e2 || 1e4 < area) continue;
		validcontours++;

		// Draw the contour
		drawContours(bw, contours, i, CV_RGB(0, 255, 0), 2, 8, hierarchy, 0);

	}

	Mat temp;
	int q = 0;

	//std::cout << "\n\n No of valid contours:  " << validcontours << endl;

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

		// Ignore if too small or too large
		if (area < 1e2 || 1e4 < area) continue;
		ValContours[i] = i;
		// Get the object orientation

		getOrientation(contours[i], bw);

		//	cout << "\n\nEig Vec X:\n" << eigen_vecs[0].x;
		//	cout << "\n\nEig Vec Y:\n" << eigen_vecs[0].y;

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

	cv::Mat top_left
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
		//cout<< "\n\nQuad1\n";
	}

	if (MaxQuadIntensity == QuadIntensity[1])
	{
		AngleDeg = ((atan2(abs(EigVecY), abs(EigVecX)) / 3.141592) * 180) - 90;
		//cout << "\n\nQuad2\n";
	}

	if (MaxQuadIntensity == QuadIntensity[2])
	{
		AngleDeg = ((atan2(abs(EigVecY), abs(EigVecX)) / 3.141592) * 180) + 90;
		//cout << "\n\nQuad3\n";
	}

	if (MaxQuadIntensity == QuadIntensity[3])
	{
		AngleDeg = ((atan2((-1 * abs(EigVecY)), abs(EigVecX)) / 3.141592) * 180) + 90;
		//cout << "\n\nQuad4\n";
	}


	cout << "\n\n EigVecX before Rot : \n" << endl;
	cout << eigen_vecs[0].x;
	cout << "\n\n EigVecY before Rot : \n" << endl;
	cout << eigen_vecs[0].y;

	cout << "\n\n Angular Orientation : \n" << endl;
	cout << AngleDeg;

	Mat M = getRotationMatrix2D(pt, AngleDeg, 1.0);

	warpAffine(bw, dst, M, Size((3 * bw.rows), (3 * bw.cols)));

	imshow("Rotated Image", dst);
cv:waitKey(0);

	for (int i = 0; i < 2; i++)
		delete[] EigVecs[i];
	delete[] EigVecs;
	for (int i = 0; i < 2; i++)
		delete[] RoundEigVecs[i];
	delete[] RoundEigVecs;

	delete[] ValContours;

}
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
bool loadKNNDataAndTrainKNN(void) {

    // read in training classifications ///////////////////////////////////////////////////

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

                                                                                // train //////////////////////////////////////////////////////////////////////////////

                                                                                // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
                                                                                // even though in reality they are multiple images / numbers
    kNearest->setDefaultK(5);

    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PossiblePlate> detectCharsInPlates(std::vector<PossiblePlate> &vectorOfPossiblePlates) {
    int intPlateCounter = 0;				// this is only for showing steps
    cv::Mat imgContours;
    std::vector<std::vector<cv::Point> > contours;
    cv::RNG rng;

    if (vectorOfPossiblePlates.empty()) {               // if vector of possible plates is empty
        return(vectorOfPossiblePlates);                 // return
    }
    // at this point we can be sure vector of possible plates has at least one plate

    for (auto &possiblePlate : vectorOfPossiblePlates) {            // for each possible plate, this is a big for loop that takes up most of the function

        preprocess(possiblePlate.imgPlate, possiblePlate.imgGrayscale, possiblePlate.imgThresh);        // preprocess to get grayscale and threshold images

#ifdef SHOW_STEPS
     //   //cv::imshow("5a", possiblePlate.imgPlate);
     //   //cv::imshow("5b", possiblePlate.imgGrayscale);
     //   //cv::imshow("5c", possiblePlate.imgThresh);
#endif	// SHOW_STEPS

        // upscale size by 60% for better viewing and character recognition
        cv::resize(possiblePlate.imgThresh, possiblePlate.imgThresh, cv::Size(), 1.6, 1.6);

        // threshold again to eliminate any gray areas
        cv::threshold(possiblePlate.imgThresh, possiblePlate.imgThresh, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);

#ifdef SHOW_STEPS
     //   //cv::imshow("5d", possiblePlate.imgThresh);
#endif	// SHOW_STEPS

        // find all possible chars in the plate,
        // this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        std::vector<PossibleChar> vectorOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh);

        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK);
        contours.clear();

        for (auto &possibleChar : vectorOfPossibleCharsInPlate) {
            contours.push_back(possibleChar.contour);

			
        }

        cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);

   //     //cv::imshow("6", imgContours);


        // given a vector of all possible chars, find groups of matching chars within the plate
        std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInPlate = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInPlate);

	//	vectorOfPossibleCharsInPlate<<PossibleChar>>

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK);

        contours.clear();
		/*
		for (size_t i = 0; i < contours.size(); i++)
		{
			double AngleDeg=getOrientation(contours[i], imgContours);
			cout << "\n\n Angular Orientation : \n" << endl;
			cout << AngleDeg;
		}
		*/
        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {
            int intRandomBlue = rng.uniform(0, 256);
            int intRandomGreen = rng.uniform(0, 256);
            int intRandomRed = rng.uniform(0, 256);

            for (auto &matchingChar : vectorOfMatchingChars) {
                contours.push_back(matchingChar.contour);
            }
            cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
        }
       // //cv::imshow("7", imgContours);
		/*
		for (size_t i = 0; i < contours.size(); i++)
		{
			double AngleDeg = getOrientation(contours[i], imgContours);
			cout << "\n\n Angular Orientation : \n" << endl;
			cout << AngleDeg;
		}
		*/
#endif	// SHOW_STEPS

        if (vectorOfVectorsOfMatchingCharsInPlate.size() == 0) {                // if no groups of matching chars were found in the plate
#ifdef SHOW_STEPS
            std::cout << "chars found in plate number " << intPlateCounter << " = (none), click on any image and press a key to continue . . ." << std::endl;
            intPlateCounter++;
            cv::destroyWindow("8");
            cv::destroyWindow("9");
            cv::destroyWindow("10");
            cv::waitKey(0);
#endif	// SHOW_STEPS
            possiblePlate.strChars = "";            // set plate string member variable to empty string
            continue;                               // go back to top of for loop
        }

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {                                         // for each vector of matching chars in the current plate
            std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);      // sort the chars left to right
            vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                     // and eliminate any overlapping chars
        }

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK);

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {
            int intRandomBlue = rng.uniform(0, 256);
            int intRandomGreen = rng.uniform(0, 256);
            int intRandomRed = rng.uniform(0, 256);

            contours.clear();

            for (auto &matchingChar : vectorOfMatchingChars) {
                contours.push_back(matchingChar.contour);
            }
            cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
			/*
			for (size_t i = 0; i < contours.size(); i++)
			{
				double AngleDeg = getOrientation(contours[i], imgContours);
				cout << "\n\n Angular Orientation : \n" << endl;
				cout << AngleDeg;
			}*/
        }
     //   //cv::imshow("8", imgContours);
#endif	// SHOW_STEPS

        // within each possible plate, suppose the longest vector of potential matching chars is the actual vector of chars
        unsigned int intLenOfLongestVectorOfChars = 0;
        unsigned int intIndexOfLongestVectorOfChars = 0;
        // loop through all the vectors of matching chars, get the index of the one with the most chars
        for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsInPlate.size(); i++) {
            if (vectorOfVectorsOfMatchingCharsInPlate[i].size() > intLenOfLongestVectorOfChars) {
                intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsInPlate[i].size();
                intIndexOfLongestVectorOfChars = i;
            }
        }
        // suppose that the longest vector of matching chars within the plate is the actual vector of chars
        std::vector<PossibleChar> longestVectorOfMatchingCharsInPlate = vectorOfVectorsOfMatchingCharsInPlate[intIndexOfLongestVectorOfChars];

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK);

        contours.clear();

        for (auto &matchingChar : longestVectorOfMatchingCharsInPlate) {
            contours.push_back(matchingChar.contour);
        }
        cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);

   //     //cv::imshow("9", imgContours);
#endif	// SHOW_STEPS

        // perform char recognition on the longest vector of matching chars in the plate
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestVectorOfMatchingCharsInPlate);

#ifdef SHOW_STEPS
        std::cout << "chars found in plate number " << intPlateCounter << " = " << possiblePlate.strChars << ", click on any image and press a key to continue . . ." << std::endl;
        intPlateCounter++;
        cv::waitKey(0);
#endif	// SHOW_STEPS

    }   // end for each possible plate big for loop that takes up most of the function

#ifdef SHOW_STEPS
    std::cout << std::endl << "char detection complete, click on any image and press a key to continue . . ." << std::endl;
    cv::waitKey(0);
#endif	// SHOW_STEPS

    return(vectorOfPossiblePlates);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PossibleChar> findPossibleCharsInPlate(cv::Mat &imgGrayscale, cv::Mat &imgThresh) {
    std::vector<PossibleChar> vectorOfPossibleChars;                            // this will be the return value

    cv::Mat imgThreshCopy;

    std::vector<std::vector<cv::Point> > contours;

    imgThreshCopy = imgThresh.clone();				// make a copy of the thresh image, this in necessary b/c findContours modifies the image

    cv::findContours(imgThreshCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);        // find all contours in plate

    for (auto &contour : contours) {                            // for each contour
        PossibleChar possibleChar(contour);

        if (checkIfPossibleChar(possibleChar)) {                // if contour is a possible char, note this does not compare to other chars (yet) . . .
            vectorOfPossibleChars.push_back(possibleChar);      // add to vector of possible chars
        }
    }

    return(vectorOfPossibleChars);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfPossibleChar(PossibleChar &possibleChar) {
    // this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    // note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.boundingRect.area() > MIN_PIXEL_AREA &&
        possibleChar.boundingRect.width > MIN_PIXEL_WIDTH && possibleChar.boundingRect.height > MIN_PIXEL_HEIGHT &&
        MIN_ASPECT_RATIO < possibleChar.dblAspectRatio && possibleChar.dblAspectRatio < MAX_ASPECT_RATIO) {
        return(true);
    }
    else {
        return(false);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<PossibleChar> > findVectorOfVectorsOfMatchingChars(const std::vector<PossibleChar> &vectorOfPossibleChars) {
    // with this function, we start off with all the possible chars in one big vector
    // the purpose of this function is to re-arrange the one big vector of chars into a vector of vectors of matching chars,
    // note that chars that are not found to be in a group of matches do not need to be considered further
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingChars;             // this will be the return value

    for (auto &possibleChar : vectorOfPossibleChars) {                  // for each possible char in the one big vector of chars

                                                                        // find all chars in the big vector that match the current char
        std::vector<PossibleChar> vectorOfMatchingChars = findVectorOfMatchingChars(possibleChar, vectorOfPossibleChars);

        vectorOfMatchingChars.push_back(possibleChar);          // also add the current char to current possible vector of matching chars

                                                                // if current possible vector of matching chars is not long enough to constitute a possible plate
        if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) {
            continue;                       // jump back to the top of the for loop and try again with next char, note that it's not necessary
                                            // to save the vector in any way since it did not have enough chars to be a possible plate
        }
        // if we get here, the current vector passed test as a "group" or "cluster" of matching chars
        vectorOfVectorsOfMatchingChars.push_back(vectorOfMatchingChars);            // so add to our vector of vectors of matching chars

                                                                                    // remove the current vector of matching chars from the big vector so we don't use those same chars twice,
                                                                                    // make sure to make a new big vector for this since we don't want to change the original big vector
        std::vector<PossibleChar> vectorOfPossibleCharsWithCurrentMatchesRemoved;

        for (auto &possChar : vectorOfPossibleChars) {
            if (std::find(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possChar) == vectorOfMatchingChars.end()) {
                vectorOfPossibleCharsWithCurrentMatchesRemoved.push_back(possChar);
            }
        }
        // declare new vector of vectors of chars to get result from recursive call
        std::vector<std::vector<PossibleChar> > recursiveVectorOfVectorsOfMatchingChars;

        // recursive call
        recursiveVectorOfVectorsOfMatchingChars = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsWithCurrentMatchesRemoved);	// recursive call !!

        for (auto &recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) {      // for each vector of matching chars found by recursive call
            vectorOfVectorsOfMatchingChars.push_back(recursiveVectorOfMatchingChars);               // add to our original vector of vectors of matching chars
        }

        break;		// exit for loop
    }

    return(vectorOfVectorsOfMatchingChars);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PossibleChar> findVectorOfMatchingChars(const PossibleChar &possibleChar, const std::vector<PossibleChar> &vectorOfChars) {
    // the purpose of this function is, given a possible char and a big vector of possible chars,
    // find all chars in the big vector that are a match for the single possible char, and return those matching chars as a vector
    std::vector<PossibleChar> vectorOfMatchingChars;                // this will be the return value

    for (auto &possibleMatchingChar : vectorOfChars) {              // for each char in big vector

                                                                    // if the char we attempting to find matches for is the exact same char as the char in the big vector we are currently checking
        if (possibleMatchingChar == possibleChar) {
            // then we should not include it in the vector of matches b/c that would end up double including the current char
            continue;           // so do not add to vector of matches and jump back to top of for loop
        }
        // compute stuff to see if chars are a match
        double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);
        double dblAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar);
        double dblChangeInArea = (double)abs(possibleMatchingChar.boundingRect.area() - possibleChar.boundingRect.area()) / (double)possibleChar.boundingRect.area();
        double dblChangeInWidth = (double)abs(possibleMatchingChar.boundingRect.width - possibleChar.boundingRect.width) / (double)possibleChar.boundingRect.width;
        double dblChangeInHeight = (double)abs(possibleMatchingChar.boundingRect.height - possibleChar.boundingRect.height) / (double)possibleChar.boundingRect.height;

        // check if chars match
        if (dblDistanceBetweenChars < (possibleChar.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) && dblAngleBetweenChars >10 &&
            dblAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS &&
            dblChangeInArea < MAX_CHANGE_IN_AREA &&
            dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&
            dblChangeInHeight < MAX_CHANGE_IN_HEIGHT) {
            vectorOfMatchingChars.push_back(possibleMatchingChar);      // if the chars are a match, add the current char to vector of matching chars
        }
    }

    return(vectorOfMatchingChars);          // return result
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// use Pythagorean theorem to calculate distance between two chars
double distanceBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar) {
    int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
    int intY = abs(firstChar.intCenterY - secondChar.intCenterY);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// use basic trigonometry(SOH CAH TOA) to calculate angle between chars
double angleBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar) {
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
std::vector<PossibleChar> removeInnerOverlappingChars(std::vector<PossibleChar> &vectorOfMatchingChars) {
    std::vector<PossibleChar> vectorOfMatchingCharsWithInnerCharRemoved(vectorOfMatchingChars);

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
                        std::vector<PossibleChar>::iterator currentCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), currentChar);
                        // if iterator did not get to end, then the char was found in the vector
                        if (currentCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(currentCharIterator);       // so remove the char
                        }
                    }
                    else {        // else if other char is smaller than current char
                                  // look for char in vector with an iterator
                        std::vector<PossibleChar>::iterator otherCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), otherChar);
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
std::string recognizeCharsInPlate(cv::Mat &imgThresh, std::vector<PossibleChar> &vectorOfMatchingChars) {
    std::string strChars;               // this will be the return value, the chars in the lic plate

    cv::Mat imgThreshColor;

    // sort chars from left to right
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    cv::cvtColor(imgThresh, imgThreshColor, CV_GRAY2BGR);       // make color version of threshold image so we can draw contours in color on it

    for (auto &currentChar : vectorOfMatchingChars) {           // for each char in plate
        cv::rectangle(imgThreshColor, currentChar.boundingRect, SCALAR_GREEN, 2);       // draw green box around the char

        cv::Mat imgROItoBeCloned = imgThresh(currentChar.boundingRect);                 // get ROI image of bounding rect

        cv::Mat imgROI = imgROItoBeCloned.clone();      // clone ROI image so we don't change original when we resize

        cv::Mat imgROIResized;
        // resize image, this is necessary for char recognition
        cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));

        cv::Mat matROIFloat;

        imgROIResized.convertTo(matROIFloat, CV_32FC1);         // convert Mat to float, necessary for call to findNearest

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // flatten Matrix into one row

        cv::Mat matCurrentChar(0, 0, CV_32F);                   // declare Mat to read current char into, this is necessary b/c findNearest requires a Mat

        kNearest->findNearest(matROIFlattenedFloat, 3, matCurrentChar);     // finally we can call find_nearest !!!

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       // convert current char from Mat to float

        strChars = strChars + char(int(fltCurrentChar));        // append current char to full string
    }

#ifdef SHOW_STEPS
  //  //cv::imshow("10", imgThreshColor);
#endif	// SHOW_STEPS

    return(strChars);               // return result
}

