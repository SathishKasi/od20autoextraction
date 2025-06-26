// Main.h

#ifndef MY_MAIN         // used MY_MAIN for this include guard rather than MAIN just in case some compilers or environments #define MAIN already
#define MY_MAIN

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "DetectPlates.h"
#include "PossiblePlate.h"
#include "DetectChars.h"

#include<iostream>
#include<conio.h>           // may have to modify this line if not using Windows
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

#define SHOW_STEPS            // un-comment or comment this line to show steps or not

using namespace cv;
using namespace std;


// global constants ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////
int main(void);
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);


# endif	// MAIN

