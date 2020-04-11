import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def load_image(image_iter):
    # load the image and convert it to grayscale
    load_image = cv2.imread('images/'+str(image_iter)+'.jpg')
    return load_image

def set_gray_color(load_image):
    gray_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
    return gray_image

def sobel_operator(gray_image):
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray_image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray_image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    # subtract the y-gradient from the x-gradient
    gradient_image = cv2.convertScaleAbs(cv2.subtract(gradX, gradY))
    return gradient_image

def blur_image(gradient_image):
    # blur and threshold the image
    blurred = cv2.blur(gradient_image, (9, 9))
    (_, thresh_image) = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    return thresh_image

def morphology_image(thresh_image):
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed_image = cv2.dilate(closed, None, iterations=4)
    return closed_image


def draw_contours(closed_image, load_image):
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (cnts, _) = cv2.findContours(closed_image.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    main = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    main_shape = main.shape[0]
    main_shape = int(main_shape)# onvert to int
    main = main.reshape(main_shape,2)# create 2-D array with coordinates
    x1 = main.min(axis = 0)[0] #calculate coordinates of reqtangle
    y1 = main.max(axis = 0)[1]
    x2 = main.max(axis = 0)[0]
    y2 = main.min(axis = 0)[1]
    barzones_image = cv2.rectangle(load_image,(x1.astype(int),y1.astype(int)),(x2.astype(int),y2.astype(int)),(0,255,255),3)
    return barzones_image

def show_image(barzones_image):
    cv2.imshow('Barzone', barzones_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def main_algoritm():
    image_iter = 1

    while image_iter<=30:
        image = load_image(image_iter)
        image_gray = set_gray_color(image)
        morph_image = morphology_image(blur_image(sobel_operator(image_gray)))
        draw = draw_contours(morph_image, image)
        show_image(draw)
        image_iter += 1

if __name__ == '__main__':
    main_algoritm()



























