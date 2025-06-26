
#include <windows.h>
#include <iostream> 
#include <fstream>
#include "dimNoteDimRecog.h"

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

using namespace std;
using namespace cv;
using namespace dimNoteDimRecog;

int main()
{
	std::vector<DimSet> finalOutput;
	
	finalOutput = dimNoteDimRecognition();

	std::cout << "\n finalOutput.size: \n " << finalOutput.size();

	std::cout << "\n\n Image closed\n " << endl;

	ofstream myfile;
	myfile.open("Recognized_text.txt");

	for (size_t i = 0; i < finalOutput.size(); i++)
	{
		string temp = finalOutput[i].dimNoteStrings;
		myfile << "\n Recognized String:\n"; 
			myfile << finalOutput[i].dimNoteStrings;
	/*		myfile << "\nAccuracy Percentage: \n";
		myfile << finalOutput[i].accuracyPercentage;
		myfile << "\nConfidence Percentage: \n";
		myfile << finalOutput[i].confidencePercentage;*/
		myfile << "\n";
	}
	
	myfile.close();
	return 0;
}
