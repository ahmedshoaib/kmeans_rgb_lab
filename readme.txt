Requirements :
Python : 3.6
OpenCV-python
numpy

This code contains Kmeans implementation using OpenCV and numpy. We use two color spaces : RGB and CIE L*a*b*.
In the RGB space implementation,We first implement the kmeans algorithm in rgb space, then we add a variance feture to the input rgb features and remove it from the centers obtained from the algorithm. Distance is L2 norm.
In Lab, we first normalize the input RGB(BGR in cv2) by dividing the rgb values by 255 and then converting to LAB space using cv2.cvtColor() function. The distance funciton used here is CIE delta(76) which is L2 norm distance and hence the algorithm can use the kmeans implementation given by cv2. 
We use Lab space as Lab space was developed to be more similar to our visual experience.

example command : python kmeans_rgb_var.py test2.jpeg 5 3