import cv2
import numpy as np
import matplotlib.pyplot as plt

PEN_PRESSURE = 0.0


def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image, d, 50, 50)
    return image


def pressure(image):
    global PEN_PRESSURE

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:]

    cv2.imshow('Grayscale', image)

    inverted = image
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - image[x][y]

    cv2.imshow('Inversi', inverted)

    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if(inverted[x][y] > 100):
                total_intensity += inverted[x][y]
                pixel_count += 1


    average_intensity = float(total_intensity) / pixel_count

    PEN_PRESSURE = average_intensity

    print("Total Intensitas : ", total_intensity)
    print("Jumlah Piksel : ",pixel_count)
    print("Nilai Tekanan Tulisan : ", average_intensity)
    print(determine_pen_pressure(average_intensity))

    show_histogram(inverted)


def determine_pen_pressure(raw_pen_pressure):
	comment = ""
	if(raw_pen_pressure > 180.0):
		pen_pressure = 0
		comment = "Tekanan Kuat"
	elif(raw_pen_pressure < 151.0):
		pen_pressure = 1
		comment = "Tekanan Ringan"
	else:
		pen_pressure = 2
		comment = "Tekanan Sedang"
		
	return pen_pressure, comment

def show_histogram(img):
    histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
    
    plt.plot(histr) 
    plt.show() 


def start(file_name):
	global PEN_PRESSURE

	image = cv2.imread(file_name)
	pressure(image)
	PEN_PRESSURE = round(PEN_PRESSURE, 2)
	
	return [PEN_PRESSURE]
    

def main():
    #image = cv2.imread('dataset/g06-026k-s03-02.png')
    image = cv2.imread('sample_image/a.png')

    #cv2.imshow('Gambar Asli', image)

    pressure(image)

    #print(zoning(image))

    cv2.waitKey(0)
    return

main()
