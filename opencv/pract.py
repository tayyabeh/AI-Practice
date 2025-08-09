import cv2
import numpy as np
import matplotlib.pyplot as plt

cv = cv2.imread(r'data\1.jpeg')
# print(cv.shape)
resized = cv2.resize(cv,(700,500)) # resized_image = cv2.resize(source_image, (width, height))

# cv2.imshow("resized image :: ", resized)

# cropped = cv[150:220,300:700] # [y_start:y_end, x_start:x_end] y is top bottom , x is left right

# cv2.imshow("Cropped image :: ",cropped)
# cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
# cv2.rectangle(cv,(300,150),(700,220),(0,0,255),2)
# cv2.putText(cv,"eyes",(250,140),cv2.FONT_ITALIC,2,(0,255,0),2)
# cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_img = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
# print(gray_img.shape)

# _,thresh_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)

# img = np.zeros((682,1024),dtype = np.uint8)

# cv2.rectangle(img,(300,150),(700,220),(255,255,255),-1)
# result = cv2.bitwise_and(gray_img,img)
# cv2.imshow("img",result)
# cv2.imshow("First image : ",gray_img)
# cv2.imshow("Threshold image : ",thresh_img)

# Creating Digital signatures using Alpha Blending
# img = np.zeros((682,1024,3),dtype = np.uint8) for BGR img
# img = np.zeros((682,1024),dtype = np.uint8)
# cv2.putText(img,"MAKE AI",(600,600),cv2.FONT_HERSHEY_DUPLEX,2,(255,255,255),2)
# blended_img = cv2.addWeighted(gray_img,0.7,img,0.3,0)
# cv2.imshow("img",blended_img)

# Image Histograms and Color Segmentation.

cv = cv2.imread(r'data\1.jpeg')

gray_img = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)

# cv2.imshow("cv",cv)

gray_scale = cv2.imshow("gray_img",gray_img)

# # hist = cv2.calcHist([image], [channels], mask, [histSize], [ranges])
# hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
# plt.figure()
# plt.title("GrayScale Histogram")
# plt.xlabel('Bins')
# plt.ylabel("No. of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()


equalized_img = cv2.equalizeHist(gray_img)

# cv2.imshow("equalized_img",equalized_img)

# Color segmentation 
# hsv_image = cv2.cvtColor(cv, cv2.COLOR_BGR2HSV)
# hsv_img = cv2.cvtColor(cv,cv2.COLOR_BGR2HSV)
# lower_green = np.array([80,50,50])
# upper_green = np.array([110,255,255])
# mask = cv2.inRange(hsv_img,lower_green ,upper_green)
# cv2.imshow("mask",mask)
# final = cv2.bitwise_and(cv,cv,mask = mask)
# cv2.imshow("final",final)
# hsv_value = hsv_img[100,100]
# print("hsv_value : ",hsv_value)

# cv2.imshow("hsv_img",hsv_img)

M = np.float32([[1,0,100],[0,1,50]]) # Translation matrix tx = 100 , ty = 50
trans_img = cv2.warpAffine(cv,M,(1024, 682))
# cv2.imshow("trans_img",trans_img)
# transofrmation 
# a = cv.shape 
# center  = (a[1]//2,a[0]//2)
# M = cv2.getRotationMatrix2D(center ,-45 ,1)
# trnsf = cv2.warpAffine(cv, M , (1024,682))
# cv2.imshow("trnsf",trnsf)

# scaling and resizing 




cv2.waitKey(0)
cv2.destroyAllWindows()