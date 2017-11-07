'''
Version 2.6

Created by:
    -Grady Duncan, @aDroidman

Sources:
http://stackoverflow.com/questions/13211745/detect-face-then-autocrop-pictures
https://gist.github.com/astanin/3097851
'''
import cv2
import numpy as np
from PIL import Image
import glob
import os
import sys
import time

# Static
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
padding = -1
boxScale = 1

def DetectFace(image, faceCascade, returnImage=False):

    # variables
    min_size = (50, 50)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    DOWNSCALE = 4

    # Equalize the histogram
    cv2.equalizeHist(image)

    # Detect the faces
    faces = faceCascade.detectMultiScale(image)
    print(faces)
    # faces = cv2.HaarDetectObjects(image, faceCascade, cv.CreateMemStorage(0), haar_scale, min_neighbors, haar_flags, min_size)

    # If faces are found
    # if faces!=None and returnImage:
    #     for ((x, y, w, h), n) in faces:
    #
    #         # Convert bounding box to two CvPoints
    #         pt1 = (int(x), int(y))
    #         pt2 = (int(x + w), int(y + h))
    #         cv2.rectangle(image, pt1, pt2, (255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces

def pil2cvGrey(pil_im):
    pil_im = pil_im.convert('L')
    cv_im = np.array(pil_im)
    print(cv_im.shape)
    return cv_im

def imgCrop(image, cropBox, padding):

    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]
    # Calculate scale factors
    xPadding = max(cropBox[2] * (boxScale - 1), int(padding))
    yPadding = max(cropBox[3] * (boxScale - 1), int(padding))

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box = [cropBox[0] - xPadding, cropBox[1] - yPadding, cropBox[0] + cropBox[2] + xPadding, cropBox[1] + cropBox[3] + yPadding]

    return image.crop(PIL_box)

def Crop(imagePattern, outputimg, padding, webCheck):
    paddingCheck = True
    imgList = glob.glob(imagePattern)
    img_num = 0
    print(len(imgList))
    while paddingCheck:
        if len(imgList) <= 0:
            return
        else:
            # Crop images
            for img in imgList:
                img_num += 1
                print(img_num)
                if img_num == len(imgList):
                    sys.exit()
                pil_im = Image.open(img)
                cv_im = pil2cvGrey(pil_im)
                faces = DetectFace(cv_im, faceCascade)
                print('face number:', len(faces))
                if len(faces)!=0:
                    n = 1
                    for face in faces:
                        print(face)
                        croppedImage = imgCrop(pil_im, face, padding)
                        (fname, ext) = os.path.splitext(img)
                        fname = os.path.basename(fname)
                        croppedImage.save(outputimg + '\\' + fname + ' -c' + ext)
                        n += 1
                    print('Cropping:', fname)
                else:
                    print('No faces found:', img)
                    print('Closing application')
                    time.sleep(.4)
                    continue
        # Send only if capturing from webcam            
        if webCheck:
            return

def CropSetup(padding, webCheck):
    inputimg = '/Users/zhuxinyue/ML/SFEW/val/neutral/'

    # Input folder check
    if not os.path.exists(inputimg):
        print('Input Folder not found')
    outputimg = '/Users/zhuxinyue/ML/SFEW/val2/neutral/'

    # Create output folder if missing
    if not os.path.exists(outputimg):
        os.makedirs(outputimg)

    # Get padding for crop
    while padding < 0:
        padding = -1
        break

    # Sent to Crop function
    Crop(inputimg + '*.png', outputimg, padding, webCheck)
    Crop(inputimg + '*.jpg', outputimg, padding, webCheck)

def main():
    webCheck = False
    CropSetup(padding, webCheck)
main()