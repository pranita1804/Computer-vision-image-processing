# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
from t1 import show_image,match_descriptors
import json

def overlap(img1,img2,i,j):
    sift = cv2.SIFT_create(100)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    match = match_descriptors(descriptors_1, descriptors_2)
    print(i,j,len(match))
    if len(match) > 100*0.2:
        return True
    else:
        return False

def stitch_background(img1, img2):
    sift = cv2.SIFT_create(500)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    match = match_descriptors(descriptors_1,descriptors_2)
    image1_kp = np.float32([keypoints_1[m].pt for m in match])
    image2_kp = np.float32([keypoints_2[match[m]].pt for m in match])
    M, mask = cv2.findHomography(image1_kp, image2_kp, cv2.RANSAC, 5.0)
    pt1_ = cv2.perspectiveTransform(np.float32([[0,0],[0,img1.shape[0]],[img1.shape[1],img1.shape[0]],[img1.shape[1],0]]).reshape(-1,1,2), M)
    pts = np.concatenate((np.float32([[0,0],[0,img2.shape[0]],[img2.shape[1],img2.shape[0]],[img2.shape[1],0]]).reshape(-1,1,2), pt1_), axis=0)
    [x_min, y_min] = np.int32(pts.min(axis=0).flatten())
    [x_max, y_max] = np.int32(pts.max(axis=0).flatten())
    Htrans = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]]) # translate
    stitched_img = cv2.warpPerspective(img1, Htrans.dot(M), (x_max-x_min, y_max-y_min))    
    stitched_img[-y_min:-y_min+img2.shape[0], -x_min:-x_min+img2.shape[1]] = img2
    return stitched_img 

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    overlap_arr = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                overlap_arr[i][j] = 1
            else:
                if overlap(imgs[i],imgs[j],i,j):
                    overlap_arr[i][j] = overlap_arr[j][i] = 1
                else:
                    overlap_arr[i][j] = overlap_arr[j][i] = 0
    img = stitch_background(imgs[1],imgs[0])
    for i in range(1,N-1):
            img = stitch_background(img, imgs[i+1])
    show_image(img)
    cv2.imwrite(savepath, img)
    return overlap_arr
if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
