Requirements :
Python : 3.6
OpenCV-python
numpy

This code contains Kmeans implementation using OpenCV and numpy. We use two color spaces : RGB and CIE L*a*b*.
In the RGB space implementation,We  implement the kmeans algorithm in rgb space and add an optional feature of adding the X locations, Y locations or standard deviation values corresponding to each pixel location to the feature vector for k-means processing. We later strip the additional feature columns to show the resultant image from just the rgb values. Distance is L2 norm.

In Lab, we first normalize the input RGB(BGR in cv2) by dividing the rgb values by 255 and then converting to LAB space using cv2.cvtColor() function. The distance funciton used here is CIE delta(76) which is L2 norm distance and hence the algorithm can use the kmeans implementation given by cv2. 
We use Lab space as Lab space was developed to be more similar to our visual experience.

example command : python kmeans_rgb_var.py test2.jpeg 7 3 001

program takes four additional arguments : 
 1. input image name 
 2. K
 3. W : dimension of filter for std dev
 4. option for feature selection(X,Y,std. deviation/texture) : 
 	|X  |Y  |S  |
 	|0/1|0/1|0/1|
