import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

i = 1
while i<=30:
    # load the image and convert it to grayscale
    image = cv2.imread('images/'+str(i) +'.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)


    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    main = sorted(cnts, key = cv2.contourArea, reverse = True)[0]


    main_shape = main.shape[0]
    main_shape = int(main_shape)# onvert to int
    main = main.reshape(main_shape,2)# create 2-D array with coordinates


    x1 = main.min(axis = 0)[0] #calculate coordinates of reqtangle
    y1 = main.max(axis = 0)[1]
    x2 = main.max(axis = 0)[0]
    y2 = main.min(axis = 0)[1]

    barzones = cv2.rectangle(image,(x1.astype(int),y1.astype(int)),(x2.astype(int),y2.astype(int)),(0,255,255),3)
    cv2.imshow('Barzone',barzones)


    cv2.waitKey(1000)

    cv2.destroyAllWindows()
    i+=1
    #cv2.imwrite('photo/out.png',barzones) #save image