import numpy as np
import cv2
import sys

def std_calc(img, w):                             #function to calculate std values for each pixel location
	stdMatrix = np.zeros(img.shape,np.uint8)
	ny = len(img)
	nx = len(img[0])


	for i in range(w,nx-w):
	    for j in range(w,ny-w):

       		sampleframe = img[j-w:j+w, i-w:i+w]
       		std    = np.std(sampleframe)
       		stdMatrix[j][i] = int(std)
	return stdMatrix 

if len(sys.argv) < 5:
	print("Error: Insufficient arguments, program takes four additional arguments : \n 1. input image name\n 2. K\n 3. W : dimension of filter for std dev\n 4. option for feature selection(X,Y,std/texture) : \n|X  |Y  |V  |\n|0/1|0/1|0/1|\n Eg : python kmeans_rgb_var.py test2.jpeg 5 3 101")
	sys.exit()
else:
	file = sys.argv[1]
	K = int(sys.argv[2])
	if K < 3:
		print("Error: K has to be greater than 2")
		sys.exit()
	w = int(sys.argv[3])  #dimension of filter for std dev
	if w < 1:
		print("Error: w has to be greater than 0")
		sys.exit()
	option = sys.argv[4]
	if(len(option) != 3):
		print("Error:Std option has to be of length 3")
		sys.exit()


img = cv2.imread(file) #read image
Z = np.float32(img)  
flag = 0
stdMatrix = std_calc(Z,w)   #calculate std dev values
flat_std = np.zeros((img.shape[0]*img.shape[1],1))
flat_x = np.zeros((img.shape[0]*img.shape[1],1))
flat_y = np.zeros((img.shape[0]*img.shape[1],1))

for i in range(stdMatrix.shape[0]):                   #flatten the std dev matrix. It has the same shape as image. but std dev values for r,g,b are same
	for j in range(stdMatrix.shape[1]):
		flat_std[flag] = [stdMatrix[i][j][0]]
		flag += 1


flag=0
for i in range(stdMatrix.shape[0]):                   #flatten the x-values matrix. It has the same shape as image. but std dev values for r,g,b are same
	for j in range(stdMatrix.shape[1]):
		flat_x[flag] = [i]
		flag += 1


flag = 0
for i in range(stdMatrix.shape[0]):                   #flatten the y-values matrix. It has the same shape as image. but std dev values for r,g,b are same
	for j in range(stdMatrix.shape[1]):
		flat_y[flag] = [j]
		flag += 1




'''  shows that all three values of std dev matrix corresponding to a pixel location are the same
print(stdMatrix.shape)
print(stdMatrix[100][32].shape,stdMatrix[100][32])
print("Z : 1 : ",Z.shape)
for i in range(stdMatrix.shape[0]):
	for j in range(stdMatrix.shape[1]):
		if stdMatrix[i][j][0] != stdMatrix[i][j][1] != stdMatrix[i][j][2]:
			print(stdMatrix[i][j],i,j)
			flag = 1

if flag == 1:
	print("instance")
'''




Z2 = Z.reshape((-1,3))  #flattens the rgb features dim : (h*w,3)


# concat x/y/standard deviaiton features as requested by user to the rgb feature array
if(option[0] == '1'):
	Z2 = np.concatenate((Z2,np.float32(flat_x)),axis = 1)
if(option[1] == '1'):
	Z2 = np.concatenate((Z2,np.float32(flat_y)),axis = 1)
if(option[2] == '1'):
	Z2 = np.concatenate((Z2,np.float32(flat_std)),axis = 1)




# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print("Calculting Centers ......\n")
ret,label,center=cv2.kmeans(Z2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

print("Center matrix : \n",center)
# Now convert back into uint8, and make original image
center = np.uint8(np.hsplit(center,np.array([3,4]))[0])   #remove x/y/std values from data

#print("Centers\n : ",center)

res = center[label.flatten()]
res2 = res.reshape((img.shape))


cv2.imwrite(sys.argv[0]+"out(rgb)_"+str(K)+"_"+file, res2)



##################################################################################################
########################################################################################################
################################################################################################
# Comparing with Lab color space


Z1 = Z/255.0

Z2 = cv2.cvtColor(Z1,44)
Z3 = Z2.reshape((-1,3))
#print(Z3.shape)


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print("Calculting Centers for Lab space ......\n")
ret,label,center=cv2.kmeans(Z3,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)  #distance function for CIE Lab : E76 similar to euclidean distance


print("Centers in Lab space : \n",center)
center2 = (cv2.cvtColor(center.reshape((K,1,3)),56))*255.0  #convert center coordinates to rgb
print("Centers in rgb space based on lab k-means : \n",center2)
# Now convert back into uint8, and make original image
center3 = np.uint8(center2)
res = center3[label.flatten()]

#print(res.shape)
res2 = res.reshape((img.shape))
#print(res2.shape)


cv2.imwrite(sys.argv[0]+"out(lab)_"+str(K)+"_"+file, res2)
