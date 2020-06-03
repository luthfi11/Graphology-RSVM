import cv2
import numpy as np
import matplotlib.pyplot as plt

def pressure(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Grayscale', image)

    inverse = cv2.bitwise_not(image)

    median = cv2.medianBlur(inverse, 3)
    #cv2.imshow('Median Filter', median)

    ret, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    if thresh.shape[1] > 1000:
        width = 800
    else:
        width = thresh.shape[1]

    wpercent = (width/float(thresh.shape[1]))
    hsize = int((float(thresh.shape[0])*float(wpercent)))

    res = cv2.resize(thresh, dsize=(width, hsize), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Otsu Thresholding', res)

    beforeThinning = getIntensity(res)
    print("Before Thinning :",beforeThinning)

    thinned = cv2.ximgproc.thinning(res)
    cv2.imshow('Thinned', thinned)

    afterThinning = getIntensity(thinned)
    print("After Thinning :",afterThinning)

    diff = abs(afterThinning - beforeThinning)
    print("Difference : ", diff)

    percent = (afterThinning / beforeThinning) * 100
    print("Percentage : ", percent)
    


def getIntensity(image):
    h, w = image.shape[:]

    total_intensity = 0
    for x in range(h):
        for y in range(w):
            if(image[x][y] > 0):
                total_intensity += 1

    return total_intensity


def main():
    img = cv2.imread('dataset/g06-026h-s03-01.png')

    #image = cv2.imread('sample_image/heavy.jpg')

    #cv2.imshow('Gambar Asli', image)

    pressure(img)

    #print(zoning(image))

    cv2.waitKey(0)
    return

main()
