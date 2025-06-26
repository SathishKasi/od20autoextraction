#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int, char** argv)
{
	// Load the image
	Mat src = imread(argv[1]);
	// Check if image is loaded fine
	if (!src.data)
		cerr << "Problem loading image!!!" << endl;
	// Show source image
	//imshow("src", src);
	waitKey(0);
	// Transform source image to gray if it is not
	Mat gray;
	if (src.channels() == 3)
	{
		cvtColor(src, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = src;
	}
	// Show gray image
	//imshow("gray", gray);
	waitKey(0);
	// Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
	Mat bw;
	adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	// Show binary image
	//imshow("binary", bw);
	waitKey(0);
	// Create the images that will use to extract the horizontal and vertical lines
	Mat horizontal = bw.clone();
	Mat vertical = bw.clone();
	Mat horVer = bw.clone();
	// Specify size on horizontal axis
	int horizontalsize = horizontal.cols / 30;
	// Create structure element for extracting horizontal lines through morphology operations
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
	// Apply morphology operations
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	// Show extracted horizontal lines
	//imshow("horizontal", horizontal);
	waitKey(0);
	// Specify size on vertical axis
	int verticalsize = vertical.rows / 30;
	// Create structure element for extracting vertical lines through morphology operations
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
	// Apply morphology operations
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));
	// Show extracted vertical lines
	//("vertical", vertical);
	waitKey(0);
	//addWeighted(vertical, 0.5, horizontal, 0.5, 0.0, horVer);
	horVer = vertical + horizontal;
	subtract(bw, horVer, horVer);
	//horVer = horVer + bw;
	/*
	// initialize the output matrix with zeros
	Mat new_image = Mat::zeros(horVer.size(), horVer.type());

	// create a matrix with all elements equal to 255 for subtraction
	Mat sub_mat = Mat::ones(horVer.size(), horVer.type()) * 255;

	//subtract the original matrix by sub_mat to give the negative output new_image
	subtract(sub_mat, horVer, new_image);
	*/
	bitwise_not(horVer, horVer);
	imwrite("horizontal_vertical.jpg", horVer);
	// Inverse vertical image
	vertical = horVer;
	bitwise_not(vertical, vertical);
	//imshow("vertical_bit", vertical);
	waitKey(0);
	// Extract edges and smooth image according to the logic
	// 1. extract edges
	// 2. dilate(edges)
	// 3. src.copyTo(smooth)
	// 4. blur smooth img
	// 5. smooth.copyTo(src, edges)
	// Step 1
	Mat edges;
	adaptiveThreshold(vertical, edges, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
	//imshow("edges", edges);
	waitKey(0);
	// Step 2
	Mat kernel = Mat::ones(2, 2, CV_8UC1);
	dilate(edges, edges, kernel);
	//imshow("dilate", edges);
	waitKey(0);
	// Step 3
	Mat smooth;
	vertical.copyTo(smooth);
	// Step 4
	blur(smooth, smooth, Size(2, 2));
	// Step 5
	smooth.copyTo(vertical, edges);
	// Show final result
	//imshow("smooth", vertical);
	adaptiveThreshold(vertical, vertical, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
	bitwise_not(vertical, vertical);
	imwrite("vertical.jpg", vertical);

	Mat whitemat = Mat::ones(vertical.size(), vertical.type()) * 255;
	vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	HoughCircles(vertical, circles, CV_HOUGH_GRADIENT, 1, vertical.rows / 8, 200, 150, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle outline
		circle(whitemat, center, radius, Scalar(0, 0, 0), 6, 8, 0);
	}

	/// Show your results
	
	imwrite("Hough Circle Transform Demo.jpg", whitemat);


	waitKey(0);
	return 0;
}