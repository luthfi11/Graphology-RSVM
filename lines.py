import cv2
import numpy as np
from matplotlib import pyplot as plt


def check_zones(separators):
    print(separators)
    upperZoneSize = separators[1] - separators[0]
    middleZoneSize = separators[2] - separators[1]
    lowerZoneSize = separators[3] - separators[2]

    print(upperZoneSize, middleZoneSize, lowerZoneSize)

    dominanceMinimum = 0.7
    print(dominanceMinimum*(middleZoneSize + lowerZoneSize))
    print(dominanceMinimum*(lowerZoneSize + upperZoneSize))
    print(dominanceMinimum*(upperZoneSize + middleZoneSize))

    zoneDominance = ""
    if upperZoneSize > dominanceMinimum*(middleZoneSize + lowerZoneSize):
        zoneDominance = "Upper Zone Dominant"
    elif lowerZoneSize > dominanceMinimum*(upperZoneSize + middleZoneSize):
        zoneDominance = "Lower Zone Dominant"
    else:
        zoneDominance = "Middle Zone Dominant"

    return zoneDominance

#img = cv2.imread('sample_image/upperZone.jpg')
img = cv2.imread('sample_dataset/g06-042i-s01-02.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:]
inverted = gray
for x in range(h):
    for y in range(w):
        inverted[x][y] = 255 - gray[x][y]

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

cv2.imshow('.jpg', thresh)

(h, w) = thresh.shape[:2]

asas = []
for j in range(h):
    row = thresh[j:j+1, 0:w]
    asas.append(np.sum(row))

higher = max(asas)/2
lits2 = [i for i in asas if i >= higher] 
topMiddleZone = (asas.index(lits2[0]))
bottomMiddleZone = (asas.index(lits2[len(lits2)-1]))

sumRows = []
for j in range(h):
    row = thresh[j:j+1, 0:w]
    sumRows.append(np.sum(row))
    if np.sum(row) > 0:
        break

ss = []
for j in range(h,-1,-1):
    row = thresh[j:j+1, 0:w]
    ss.append(np.sum(row))
    if np.sum(row) > 0:
        break
    
cv2.line(img, (0, len(sumRows)), (w, len(sumRows)), (0, 255, 0), 2)
cv2.line(img, (0, img.shape[0]-len(ss)), (w, img.shape[0]-len(ss)), (0, 255, 0), 2)
cv2.line(img, (0, topMiddleZone), (w, topMiddleZone), (0, 255, 0), 2)
cv2.line(img, (0, bottomMiddleZone), (w, bottomMiddleZone), (0, 255, 0), 2)
cv2.imshow('houghlines3.jpg',img)

separators = [len(sumRows), topMiddleZone, bottomMiddleZone, img.shape[0]-len(ss)]
print(check_zones(separators))


cv2.waitKey(0)