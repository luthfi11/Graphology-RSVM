import matplotlib.pyplot as plt
import cv2
import numpy as np

image = cv2.imread('sample_image/graying.jpg')
#cv2.imshow("Asli", image)

for i in range(50):
    R = image[0][i][2]
    G = image[0][i][1]
    B = image[0][i][0]
    hasil = round((0.2989 * R) + (0.5870 * G) + (0.1141 * B))
    #print(i+1,".(0.2989 *",R,") + (0.5870 *",G,") + (0.1141 *",B,") =",hasil)

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Grayscale", grayscale)

#h, otsu = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imshow("Otsu Thresholding", otsu)

h, w = grayscale.shape[:]

# Image Inversion
inverted = np.copy(grayscale)
for i in range(50):
    has = 255 - grayscale[0][i]
    #print(i+1,". 255 -",grayscale[0][i],"=",has)


inverse = cv2.bitwise_not(grayscale)
# print(*inverse[0])
#cv2.imshow("Inverse", inverse)
#cv2.imwrite("sample_image/inverse.png", inverse)


med = cv2.medianBlur(inverse, 3)
#from skimage.morphology import disk
#from skimage.filters import median
#med = median(inverse, disk(3))

# print(*med[0])
#cv2.imshow("Median", median)
#cv2.imwrite("sample_image/median.png", median)

h, thresh = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Otsu", thresh)
#cv2.imwrite("sample_image/threshold.png", thresh)

#dilasi = cv2.dilate(thresh, (5,5), iterations=2)
#cv2.imshow("Dilasi", dilasi)

h, w = thresh.shape[:]
sumRows = []
pixelRow = []
for j in range(h):
    row = thresh[j:j+1, 0:w]
    sumRows.append(np.sum(row))
    pixelRow += [j]

plt.barh(pixelRow, sumRows, height=1)
plt.ylim(len(pixelRow), 0)
plt.ylabel("Baris Citra")
plt.xlabel("Jumlah Intensitas Piksel")
plt.show()

cv2.waitKey(0)
