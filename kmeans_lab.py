import numpy as np
import cv2
import sys

file = sys.argv[1]
K = int(sys.argv[2])
img = cv2.imread(file)


# convert to np.float32
Z = np.float32(img)

Z1 = Z/255.0

Z2 = cv2.cvtColor(Z1,44)
Z3 = Z2.reshape((-1,3))



# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z3,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
res2 = cv2.cvtColor(res2,56)
res3 = res2*255.0

cv2.imwrite(sys.argv[0]+"out_"+str(K)+"_"+file, res2)