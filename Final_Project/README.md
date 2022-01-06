# CV-Final-Project
Computer Vision Social Distancing Final Project

https://github.com/endernac/CV-Final-Project

There are two main components to this project:

## User-based homography selection
This is intended to be run locally, since Google Colaboratory does not support XWindows. This notebook uses the OpenCV `imshow` with a few modifications to allow for simple homography selection. The user should select 4 points in the image which represent a square on the ground plane. The order of selection should be:
1. bottom left
2. bottom right
3. top right
4. top left

Double clicking on a point in the image will create a dot, indicating the selected location. Once four points have been selected, the user can double click near points to shift them. Additionally, a green grid will be displayed to help guide selection. This grid may be difficult to work with at first, as poor selections will make strange vanishing points. However, after adjustments are made, an ideal selection will make it very clear where the ground plane is. When the desired grid has been selected, hit ESC to return. The homography of interest will be displyed in the jupyter notebook, which can be copied (with comas added) into the Colaboratory notebook.

## Detection pipeline
The following notebooks are meant to be run on Google Colaboratory. Two notebooks are provided, preloaded with information for a simple run of some VIRAT example data. `people_detector.ipynb` draws circles on an overhead view without any indication of violating social distance while `people_detector_distancing_visualization.ipynb` changes circle color based on violations. These notebooks are meant to be run on Google Colaboratory to leverage the GPUs provided. 

These run with a basic pipeline: 
1. Homography loading (from corresponding VIRAT file, or manualy input after selection from computer)
2. Detection loop (with calibration during the first frame with people)

Currently, the camera pose calculation is set up such that the angle under 90 degrees should be chosen. 

