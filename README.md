This repository contains an Optical Character Recognition (OCR) system specifically designed for dimensional annotation recognition in engineering drawings. The system is built using OpenCV and C++.

Repository Structure
Main Components
1. Non_UI_Based/dimNoteDimRecog/

Core OCR engine for dimension and note recognition
Contains the main algorithm in
Performs multiple iterations of image processing (horizontal, vertical, rotated orientations)
Uses machine learning (KNN) for character recognition
2. UI_Based/dimNoteDimRecog/

User interface version of the same OCR functionality
Allows interactive processing of engineering drawings
3. OpenCV_Sources/

Reference implementations including:
KNN Character Recognition system
License plate recognition (adapted for engineering drawings)
Training and testing utilities
4. Input_Images/ & Results/

Sample engineering drawings (PDF, TIF, PNG formats)
Processed output images showing recognized dimensions
Key Functionality
The system processes engineering drawings through several stages:

Image Preprocessing: Converts drawings to grayscale, applies thresholding
Line Detection: Uses Hough transforms to detect and remove construction lines
Circle Detection: Identifies circular annotations and dimensions
Text Recognition: Employs KNN classification to recognize dimensional text
Multi-orientation Processing: Analyzes images in 4 different orientations for comprehensive recognition
Technical Features
Skew Correction: Automatically corrects rotated drawings
Morphological Operations: Cleans up images for better character recognition
Contour Analysis: Identifies text regions and dimensional annotations
Machine Learning: Uses trained models (, ) for character recognition

