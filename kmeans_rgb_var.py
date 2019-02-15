import numpy as np
import cv2
import sys

def variance_calc(img, w):                             #function to calculate variance values for each pixel location
	varianceMatrix = np.zeros(img.shape,np.uint8)
	ny = len(img)
	nx = len(img[0])


	for i in range(w,nx-w):
	    for j in range(w,ny-w):

       		sampleframe = img[j-w:j+w, i-w:i+w]
       		variance    = np.var(sampleframe)
       		varianceMatrix[j][i] = int(variance)
	return varianceMatrix 

file = sys.argv[1]
K = int(sys.argv[2])
w = int(sys.argv[3])  #dimension of filter for variance
img = cv2.imread(file) #read image
Z = np.float32(img)  
flag = 0
varianceMatrix = variance_calc(Z,w)   #calculate variance values
flat_var = np.zeros((img.shape[0]*img.shape[1],1))

for i in range(varianceMatrix.shape[0]):                   #flatten the variance matrix. It has the same shape as image. but variance values for r,g,b are same
	for j in range(varianceMatrix.shape[1]):
		flat_var[flag] = varianceMatrix[i][j][0]
		flag += 1
#print(flat_var.shape)


'''  shows that all three values of Variance matrix corresponding to a pixel location are the same
print(varianceMatrix.shape)
print(varianceMatrix[100][32].shape,varianceMatrix[100][32])
print("Z : 1 : ",Z.shape)
for i in range(varianceMatrix.shape[0]):
	for j in range(varianceMatrix.shape[1]):
		if varianceMatrix[i][j][0] != varianceMatrix[i][j][1] != varianceMatrix[i][j][2]:
			print(varianceMatrix[i][j],i,j)
			flag = 1

if flag == 1:
	print("instance")
'''
Z2 = Z.reshape((-1,3))
#print("Z : 2 : ",Z2.shape)



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imwrite(sys.argv[0]+"out(rgb)_"+str(K)+"_"+file, res2)



Z3 = np.concatenate((Z2,np.float32(flat_var)),axis = 1)  #include corresponding variance values in input data
#print(Z3.shape)
#print(Z3[0],Z3[1])




# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z3,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(np.hsplit(center,np.array([3,4]))[0])   #remove variance values from data

#print("Centers\n : ",center)

res = center[label.flatten()]
res2 = res.reshape((img.shape))


cv2.imwrite(sys.argv[0]+"out(rgb+var)_"+str(K)+"_"+file, res2)

# Comparing with Lab color space


Z1 = Z/255.0

Z2 = cv2.cvtColor(Z1,44)
Z3 = Z2.reshape((-1,3))
#print(Z3.shape)


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z3,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)  #distance function for CIE Lab : E76 similar to euclidean distance
#print(type(ret),label.shape,center.shape)
#print("ret : ",ret)
#print(center)

center2 = (cv2.cvtColor(center.reshape((K,1,3)),56))*255.0  #convert center coordinates to rgb

# Now convert back into uint8, and make original image
center3 = np.uint8(center2)
res = center3[label.flatten()]

#print(res.shape)
res2 = res.reshape((img.shape))
#print(res2.shape)


cv2.imwrite(sys.argv[0]+"out(lab)_"+str(K)+"_"+file, res2)