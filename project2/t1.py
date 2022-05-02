#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def match_descriptors(descriptors_1, descriptors_2):
    match= {}
    for i in range(len(descriptors_1)):
        ssd = {}
        for j in range(len(descriptors_2)):
            dist = sum(np.square(descriptors_1[i] - descriptors_2[j]))
            ssd[j] = dist
        ssd = dict(sorted(ssd.items(), key=lambda item: item[1]))
        first_value = list(ssd.values())[0]
        second_value = list(ssd.values())[1]
        if first_value/second_value < 0.8:
            first_key = list(ssd.keys())[0]
            match[i] = first_key
    return match 


def stitch(image1_kp, image2_kp, img1, img2):
    M, mask = cv2.findHomography(image1_kp, image2_kp, cv2.RANSAC, 5.0)
    result = cv2.warpPerspective(img1, M, ((img1.shape[1] + img2.shape[1]), img1.shape[0] + img2.shape[0]))
    for i in range(0, img2.shape[0]):
        for j in range(0, img2.shape[1]):
            if np.sum(result[i][j]) > 0:
                if np.sum(result[i][j]) > np.sum(img2[i][j]):
                    result[i][j] = result[i][j]
                else:
                    result[i][j] = img2[i][j]
            else:
                result[i][j] = img2[i][j]
    return result


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    sift = cv2.SIFT_create(500)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    match = match_descriptors(descriptors_1, descriptors_2)
    image1_kp = np.float32([keypoints_1[m].pt for m in match])
    image2_kp = np.float32([keypoints_2[match[m]].pt for m in match])
    image = stitch(image1_kp, image2_kp,img1, img2)
    cv2.imwrite(savepath, image)
    
    
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

