import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology


def findZone(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:]

    # Median Filter
    median = cv2.medianBlur(gray, 3)
    #cv2.imshow('Median Filter', median)

    # Otsu Thresholding
    ret, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    #cv2.imshow('Otsu Thresholding', thresh)

    # Count all pixel rows
    sumRows = []
    pixelRow = []
    for j in range(h):
        row = thresh[j:j+1, 0:w]
        pixel = np.count_nonzero(row == 0)
        sumRows += [pixel]
        pixelRow += [j]

    # Search for first stroke
    topRow = []
    for j in range(h):
        row = thresh[j:j+1, 0:w]
        topRow.append(np.sum(row))
        pixel = np.count_nonzero(row == 0)
        if pixel > 0:
            break

    # Search for last stroke
    bottomRow = []
    for j in range(h, -1, -1):
        row = thresh[j:j+1, 0:w]
        bottomRow.append(np.sum(row))
        pixel = np.count_nonzero(row == 0)
        if pixel > 0:
            break

    # Search middle lines
    mostLines = max(sumRows)/3
    morePixel = [i for i in sumRows if i >= mostLines]

    # Define all zone lines
    topZone = len(topRow)-1
    topMiddleZone = sumRows.index(morePixel[0])
    bottomMiddleZone = sumRows.index(morePixel[len(morePixel)-1])
    bottomZone = img.shape[0] - len(bottomRow) + 1

    # Generate Lines
    cv2.line(img, (0, topZone), (w, topZone), (0, 255, 0), 2)
    cv2.line(img, (0, topMiddleZone), (w, topMiddleZone), (0, 255, 0), 2)
    cv2.line(img, (0, bottomMiddleZone), (w, bottomMiddleZone), (0, 255, 0), 2)
    cv2.line(img, (0, bottomZone), (w, bottomZone), (0, 255, 0), 2)

    #cv2.imshow('Zone Lines', img)
    #cv2.imwrite('sample_image/zoned.png', img)

    #show_histogram(sumRows, pixelRow)

    separators = [topZone, topMiddleZone, bottomMiddleZone, bottomZone]
    
    top = separators[1] - separators[0]
    middle = separators[2] - separators[1]
    bottom = separators[3] - separators[2]
    
    return top, middle, bottom

def show_histogram(img, label):
    plt.barh(label, img, height=1)
    plt.ylim(len(label),0)
    plt.ylabel("Baris Citra (y)")
    plt.xlabel("Jumlah Intensitas Piksel")
    plt.show()

def extract(file_name):
    image = cv2.imread(file_name)
    top, middle, bottom = findZone(image)
	
    return [top, middle, bottom]

def main():
    img = cv2.imread('dataset/h06-089-s01-00.png')
    
    findZone(img)
