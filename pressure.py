import cv2
import numpy as np

PEN_PRESSURE = 0.0


def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image, d, 50, 50)
    return image


def pressure(image):
    global PEN_PRESSURE

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:]

    inverted = image
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - image[x][y]

    cv2.imshow('Inversi', inverted)

    filtered = bilateralFilter(inverted, 3)
    cv2.imshow('Bilateral', filtered)

    ret, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)
    cv2.imshow('Threshold', thresh)

    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if(thresh[x][y] > 0):
                total_intensity += thresh[x][y]
                pixel_count += 1

    average_intensity = float(total_intensity) / pixel_count

    PEN_PRESSURE = average_intensity

    print(total_intensity)
    print(pixel_count)
    print("Average pen pressure: "+str(average_intensity))

    #print(determine_pen_pressure(average_intensity))

    return

def determine_pen_pressure(raw_pen_pressure):
	comment = ""
	if(raw_pen_pressure > 180.0):
		pen_pressure = 0
		comment = "HEAVY"
	elif(raw_pen_pressure < 151.0):
		pen_pressure = 1
		comment = "LIGHT"
	else:
		pen_pressure = 2
		comment = "MEDIUM"
		
	return pen_pressure, comment

def start(file_name):
	global PEN_PRESSURE

	image = cv2.imread(file_name)
	pressure(image)
	PEN_PRESSURE = round(PEN_PRESSURE, 2)
	
	return [PEN_PRESSURE]

def main():
    image = cv2.imread('dataset/r06-103-s00-05.png')
    cv2.imshow('Gambar Asli', image)

    pressure(image)

    cv2.waitKey(0)
    return

main()
