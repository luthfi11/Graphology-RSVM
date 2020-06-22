import cv2
import numpy as np
import matplotlib.pyplot as plt


def pressure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:]
    
    #cv2.imshow('Grayscale', image)

    median = cv2.medianBlur(gray, 3)
    #cv2.imshow('Median Filter', median)

    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if(median[x][y] < 150):
                total_intensity += median[x][y]
                pixel_count += 1

    average_intensity = round((total_intensity / pixel_count),2)
    percentage = round(((average_intensity * 100) / 255),2)

    """
    print("Total Intensitas : ", total_intensity)
    print("Jumlah Piksel : ",pixel_count)
    print("Rata-Rata Intensitas : ", average_intensity)
    print("Persentase :",percentage)

    print(determine_pen_pressure(percentage))
    
    show_histogram(median)
    """
    return average_intensity, percentage, determine_pen_pressure(percentage)


def determine_pen_pressure(raw_pen_pressure):
	comment = ""
	if(raw_pen_pressure < 30.0):
		pen_pressure = 0
		comment = "Kuat"
	elif(raw_pen_pressure < 40.0):
		pen_pressure = 1
		comment = "Sedang"
	else:
		pen_pressure = 2
		comment = "Ringan"
		
	return comment

def show_histogram(img):
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(img.ravel(), 150, [0,150])

    ax.set_xlabel('Intensitas Piksel')
    ax.set_ylabel('Jumlah')

    fig.tight_layout()
    plt.show()


def start(file_name):
	image = cv2.imread(file_name)
	total, count, clas = pressure(image)
	
	return [total, count, clas]

def extract(file_name):
    image = cv2.imread(file_name)
    total, count, clas = pressure(image)
	
    return [total, count]

def main():
    image = cv2.imread('dataset/m06-106-s01-01.png')
    
    pressure(image)

    #print(zoning(image))

    cv2.waitKey(0)
    return

main()
