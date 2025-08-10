import cv2
import numpy as np
import matplotlib.pyplot as plt

# cv = cv2.imread(r'data\1.jpeg')
# # print(cv.shape)
# resized = cv2.resize(cv,(700,500)) # resized_image = cv2.resize(source_image, (width, height))

# # cv2.imshow("resized image :: ", resized)

# # cropped = cv[150:220,300:700] # [y_start:y_end, x_start:x_end] y is top bottom , x is left right

# # cv2.imshow("Cropped image :: ",cropped)
# # cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
# # cv2.rectangle(cv,(300,150),(700,220),(0,0,255),2)
# # cv2.putText(cv,"eyes",(250,140),cv2.FONT_ITALIC,2,(0,255,0),2)
# # cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

# # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
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

# # Image Histograms and Color Segmentation.

# cv = cv2.imread(r'data\1.jpeg')

# gray_img = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)

# # cv2.imshow("cv",cv)

# gray_scale = cv2.imshow("gray_img",gray_img)

# # hist = cv2.calcHist([image], [channels], mask, [histSize], [ranges])
# hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
# plt.figure()
# plt.title("GrayScale Histogram")
# plt.xlabel('Bins')
# plt.ylabel("No. of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()


# equalized_img = cv2.equalizeHist(gray_img)

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

# M = np.float32([[1,0,100],[0,1,50]]) # Translation matrix tx = 100 , ty = 50
# trans_img = cv2.warpAffine(cv,M,(1024, 682))
# # cv2.imshow("trans_img",trans_img)
# transofrmation 
# a = cv.shape 
# center  = (a[1]//2,a[0]//2)
# M = cv2.getRotationMatrix2D(center ,-45 ,1)
# trnsf = cv2.warpAffine(cv, M , (1024,682))
# cv2.imshow("trnsf",trnsf)

# scaling and resizing 
# cv = cv2.imread(r'data\1.jpeg')
# #scaling 
# r = cv2.resize(cv,(800,600))
# # flipped _image 
# fliped = cv2.flip(r,1)
# cv2.imshow("fliped",fliped)

# # Affine Transformations used for scanned docs 
# source_points = np.float32([[0,0],[1024,0],[0,682]])
# destination_points = np.float32([[0,0],[1024,200],[0,682]])
# M = cv2.getAffineTransform(source_points,destination_points)
# final = cv2.warpAffine(cv,M,(1024,682))
# cv2.imshow("final",final)

# import cv2 
# cap = cv2.VideoCapture(1)

# while True :
#     ret , img = cap.read()
#     cv2.imshow("img",img)  

#     if cv2.waitKey(1) &  0xFF ==  ord('q') :
#         break



# Contour Detection
# gy = cv2.cvtColor(cv , cv2.COLOR_BGR2GRAY)
# _,thresh_img = cv2.threshold(gy,110 ,255,cv2.THRESH_BINARY)
# contour , hirarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(source_image, contours, contour_index, color, thickness)
# # con_img  = cv2.drawContours(cv, contour , -1 , (0,255,0), 2)
# # area = cv2.contourArea(contour) 
# con_img = cv.copy()
# for con in contour :
#     area = cv2.contourArea(con)
#     if area >= 50000 :
#         con_img  = cv2.drawContours(con_img, [con],0 , (0,255,0), 2)
#     else:
#         continue

# Template Matching
# source = cv2.imread(r'e:\Pictures\Saved Pictures\t1.jpg')
# target = cv2.imread(r'e:\Pictures\Saved Pictures\t3.jpg')
# print(source.shape)
# print(target.shape)
# s_gray = cv2.cvtColor(source , cv2.COLOR_BGR2GRAY)
# t_gray = cv2.cvtColor(target , cv2.COLOR_BGR2GRAY)

# matched_img = cv2.matchTemplate(s_gray , t_gray , cv2.TM_CCOEFF_NORMED)


# cv2.imshow("matched image ", matched_img)

# #find the coordinates 

# min_val , max_val, min_loc , max_loc = cv2.minMaxLoc(matched_img)
# x = target.shape[1]+ max_loc[0]
# y = target.shape[0] + max_loc[1]

# endpoint = (x,y)

# img = cv2.rectangle(source , max_loc , endpoint,(0,0,255),2)
# cv2.imshow("img",img)





# # Image Pyramids
# source = cv2.imread(r'e:\Pictures\Saved Pictures\t1.jpg')
# py_down = cv2.pyrDown(source)
# py_up = cv2.pyrUp(source)
# seco_py = cv2.pyrDown(py_down)
# seco_up = cv2.pyrUp(seco_py)

# cv2.imshow("py_down",seco_py) # twice of height and width
# cv2.imshow("py_up",py_up) # half owidhand height
# cv2.imshow("seco_up",seco_up)
# cv2.imshow("compared image :: ", source)

#  Feature Detection and Description
# cv = cv2.imread(r'data\1.jpeg')
# copy_cv = cv.copy()
# gray_img = cv2.cvtColor(copy_cv, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray_img",gray_img)

# corner_img = cv2.cornerHarris(gray_img ,2,3,0.04)
# ds_norm = cv2.normalize(corner_img,None,0,255,cv2.NORM_MINMAX)
# cv2.imshow("ds_norm",ds_norm)
# img = np.uint8(ds_norm)
# corners = np.argwhere(img > 0.01*img.max())
# print(corners)
# for cor in corners :
#     cv2.circle(copy_cv,(cor[1] ,cor[0]),10,(0,255,0),2)

# cv2.imshow("final",copy_cv)

# import cv2
# import numpy as np

# # Use the file path to the image you downloaded
# cv = cv2.imread(r'data\blox.jpg')
# copy_cv = cv.copy()
# gray_img = cv2.cvtColor(copy_cv, cv2.COLOR_BGR2GRAY)

# # Apply Harris Corner Detector
# corner_img = cv2.cornerHarris(gray_img ,2,3,0.04)

# # Normalize and convert to 8-bit integer
# ds_norm = cv2.normalize(corner_img,None,0,255,cv2.NORM_MINMAX)
# img = np.uint8(ds_norm)

# # Find and draw corners
# corners = np.argwhere(img > 0.001*img.max())
# print(f"Number of corners detected: {len(corners)}")

# for cor in corners :
#     cv2.circle(copy_cv,(cor[1] ,cor[0]),5,(0,255,0),2)

# cv2.imshow("final",copy_cv)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Load your image
cv = cv2.imread(r'data\1.jpeg')

# Make a copy to draw on
copy_cv = cv.copy()

# Convert to grayscale
gray = cv2.cvtColor(copy_cv, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)

# Apply Harris Corner Detector
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# # Dilate the result to make the corners bigger
# dst = cv2.dilate(dst, None)

# # Threshold for an optimal value and draw circles on the original image
# copy_cv[dst > 0.01 * dst.max()] = [0, 0, 255]

# cv2.imshow('final', copy_cv)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Applying  Shi-Tomasi Corner Detector

# corners = cv2.goodFeaturesToTrack(gray , 50 ,0.01 , 10)
# # print(corners)
# for cor in corners :
#      cv2.circle(copy_cv,(int(cor[0][0]), int(cor[0][1])),5,(0,255,0),2)
# cv2.imshow("new",copy_cv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Applying SIFT (Scale invarient Corner Detector)
sift  = cv2.SIFT_create()
keypoints = sift.detect(gray ,None)
key_img = cv2.drawKeypoints(copy_cv,keypoints,None,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("new",key_img)
cv2.waitKey(0)
cv2.destroyAllWindows()