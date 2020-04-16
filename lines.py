import cv2
import numpy as np
from matplotlib import pyplot as plt


def setZoneClass(separators):
    upperZoneSize = separators[1] - separators[0]
    middleZoneSize = separators[2] - separators[1]
    lowerZoneSize = separators[3] - separators[2]

    dominanceMinimum = 0.7

    zoneDominance = ""
    if upperZoneSize > dominanceMinimum*(middleZoneSize + lowerZoneSize):
        zoneDominance = "Upper Zone Dominant"
    elif lowerZoneSize > dominanceMinimum*(upperZoneSize + middleZoneSize):
        zoneDominance = "Lower Zone Dominant"
    else:
        zoneDominance = "Middle Zone Dominant"

    return zoneDominance

def findZone(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:]

    # Image Inversion
    inverted = gray
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - gray[x][y]

    # Median Filter
    median = cv2.medianBlur(inverted,5)
    cv2.imshow('Median Filter', median)

    # Otsu Thresholding
    ret, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow('Thresholding', thresh)

    # Count all pixel rows
    sumRows = []
    for j in range(h):
        row = thresh[j:j+1, 0:w]
        sumRows.append(np.sum(row))

    # Search for first stroke
    topRow = []
    for j in range(h):
        row = thresh[j:j+1, 0:w]
        topRow.append(np.sum(row))
        if np.sum(row) > 0:
            break

    # Search for last stroke
    bottomRow = []
    for j in range(h,-1,-1):
        row = thresh[j:j+1, 0:w]
        bottomRow.append(np.sum(row))
        if np.sum(row) > 0:
            break
        
    # Search middle lines
    mostLines = max(sumRows)/2
    morePixel = [i for i in sumRows if i >= mostLines] 

    # Define all zone lines
    topZone = len(topRow)
    topMiddleZone = sumRows.index(morePixel[0])
    bottomMiddleZone = sumRows.index(morePixel[len(morePixel)-1])
    bottomZone = img.shape[0] - len(bottomRow)

    # Generate Lines
    cv2.line(img, (0, topZone), (w, topZone), (0, 255, 0), 2)
    cv2.line(img, (0, topMiddleZone), (w, topMiddleZone), (0, 255, 0), 2)
    cv2.line(img, (0, bottomMiddleZone), (w, bottomMiddleZone), (0, 255, 0), 2)
    cv2.line(img, (0, bottomZone), (w, bottomZone), (0, 255, 0), 2)

    cv2.imshow('Zone Lines',img)

    separators = [topZone, topMiddleZone, bottomMiddleZone, bottomZone]
    zoneClass = setZoneClass(separators)
    
    print(zoneClass)


def main():
    img = cv2.imread('sample_image/upper.jpg')
    #img = cv2.imread('sample_dataset/g06-042i-s01-02.png')

    findZone(img)


main()
cv2.waitKey(0)