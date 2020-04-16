import statistics
import cv2
import numpy as np
from PIL import Image as Img
from PIL import ImageTk


def straighten(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)
    # cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

    """
    for c in cntsSorted:
        print(c)
        cv2.drawContours(image, c, -1, (0,255,0), 3)
        cv2.imshow('image_final',image)
    """

    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90+angle)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # cv2.line(rotated, (10, 10), (250, 10), (0, 255, 0), 2)

    print("[INFO] Angle: {:.3f}".format(angle))
    # cv2.imshow("Input", image)
    # cv2.imshow("Rotated", rotated)

    # img = cv2.erode(rotated, np.ones((1, 50)))
    # cv2.imshow("Test",img)

    crop_image(rotated)
    return image


def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (kernel))

    cv2.imshow("Dilation", img_dilation)
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        if h > w or h < 20:
            continue

        roi = image[y:y+h, x:x+w]
        find_line(roi)
        cv2.imshow('Zone'+str(i), resize(roi))


def find_line(image):
    rot = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rot = cv2.bitwise_not(rot)
    rthresh = cv2.threshold(rot, 0, 255, cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(rthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    xx = []
    yy = []
    hh = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h > w or h < 20:
            continue

        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.line(image, (x, y), (x+w, y), (0, 255, 0), 2)
        #cv2.line(image, (x, y+h), (x+w, y+h), (0, 255, 0), 2)
        xx += [x]
        yy += [y]
        hh += [h]
        # area=cv2.contourArea(cnt)
        # print(area)
        # hull = cv2.convexHull(cnt)
        # cv2.drawContours(image, [hull], -1, (0, 255, 0), 1)

    if (len(yy) > 0):
        upperZoneTop = min(yy)
        middleZoneTop = int(statistics.mean(yy))
        bottomZoneTop = int(statistics.mean(yy))+int(statistics.mean(hh))
        bottomZoneBottom = int(statistics.mean(yy))+max(hh)
        if bottomZoneBottom > image.shape[0]:
            bottomZoneBottom = image.shape[0]

        cv2.line(image, (0, upperZoneTop), (image.shape[1], upperZoneTop), (0, 0, 255), 2)
        cv2.line(image, (0, middleZoneTop), (image.shape[1], middleZoneTop), (255, 0, 0), 2)
        cv2.line(image, (0, bottomZoneTop), (image.shape[1], bottomZoneTop), (0, 255, 0), 2)
        cv2.line(image, (0, bottomZoneBottom), (image.shape[1], bottomZoneBottom), (0, 255, 0), 2)

        separators = [upperZoneTop, middleZoneTop, bottomZoneTop, bottomZoneBottom]

        zone = check_zones(separators)
        print(zone)

    """
    if (len(yy) > 0):
        cv2.line(image, (0, yy[0]),(image.shape[1], yy[0]), (0, 255, 0), 2)
        cv2.line(image, (0, yy[len(yy)-1]+hh[len(hh)-1]),(image.shape[1], yy[len(yy)-1]+hh[len(hh)-1]), (0, 255, 0), 2)
    else:
        cv2.line(image, (0, 0),(image.shape[1], 0), (0, 255, 0), 2)
        cv2.line(image, (0, image.shape[0]),(image.shape[1], image.shape[0]), (0, 255, 0), 2)
    """

    #cv2.imshow('Lines', image)


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


def horizontalProjection(img):
    (h, w) = img.shape[:2]
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w]
        sumRows.append(np.sum(row))

    return sumRows


def verticalProjection(img):
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1]
        sumCols.append(np.sum(col))

    return sumCols


def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image, d, 50, 50)
    return image


def medianFilter(image, d):
    image = cv2.medianBlur(image, d)
    return image


def threshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return image


def dilate(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image


def erode(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image


BASELINE_ANGLE = 0.0


def straight(image):

    global BASELINE_ANGLE

    angle = 0.0
    angle_sum = 0.0
    contour_count = 0

    filtered = bilateralFilter(image, 3)
    # cv2.imshow('filtered',filtered)

    thresh = threshold(filtered, 128)
    # cv2.imshow('thresh',thresh)

    dilated = dilate(thresh, (5, 100))
    #cv2.imshow('dilated', dilated)

    ctrs, hier = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        if h > w or h < 20:
            continue

        roi = image[y:y+h, x:x+w]

        rect = cv2.minAreaRect(ctr)
        center = rect[0]
        angle = rect[2]

        if angle < -45.0:
            angle += 90.0

        rot = cv2.getRotationMatrix2D(center, angle, 1)

        extract = cv2.warpAffine(
            roi, rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        image[y:y+h, x:x+w] = extract
        cv2.imshow('segment no:'+str(i), extract)
        angle_sum += angle
        contour_count += 1

    mean_angle = angle_sum / contour_count
    BASELINE_ANGLE = mean_angle
    print("Average baseline angle: "+str(mean_angle))
    cv2.imshow("Cr", image)
    # crop_image(image)
    return image


def resize(img):
    if img.shape[1] > 1500:
        if img.shape[1] > img.shape[0] or img.shape[1] == img.shape[0]:
            width = 1400
        else:
            width = 700

        wpercent = (width/float(img.shape[1]))
        hsize = int((float(img.shape[0])*float(wpercent)))

        img = cv2.resize(img, dsize=(width, hsize), interpolation=cv2.INTER_CUBIC)
    return img

def main():
    image = cv2.imread('sample_image/upperZone.jpg')
    #image = cv2.imread('sample_dataset/c06-039-s01-01.png')

    #ss = straight(image)
    #cv2.imshow("Image", resize(image))

    straighten(image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    med = medianFilter(gray, 3)
    ret, thres = cv2.threshold(med, 128, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("AAA", thres)

    hor = horizontalProjection(thres)
    print(hor)
    T = int(max(hor) / 3)
    print(T)

    for i in hor:
        if(i > T):
            r2 = i
            break

    for i in reversed(hor):
        if(i > T):
            r1 = i
            break

    print(r1, r2)
    print("Upper zone = ",r2 - hor[0])
    print("Middle zone = ",r1 - r2)
    print("Lower zone = ",hor[len(hor)-1] - r1)
    """
    cv2.waitKey(0)
    return


main()
