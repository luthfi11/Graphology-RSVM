from __future__ import with_statement
from PIL import Image
import numpy as np

"""
im = Image.open("sample_image/threshold.png") #relative path to file
pixels = list(im.getdata())
width, height = im.size 
pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

np.savetxt("pixel_data_threshold.csv", pixels, delimiter=",") 
"""


import cv2
import matplotlib.pyplot as plt

img = cv2.imread("sample_image/graying.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(img, 3)
#cv2.imwrite("sample_image/median.png", median)
#print(*np.unique(median), sep='\n')

t, th = cv2.threshold(median,0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
#cv2.imwrite("sample_image/threshold.png", th)
"""
fig, ax = plt.subplots()
n, bins, patches = ax.hist(median.ravel(), 256, [0,255], density=1)
ax.set_xlabel('Intensitas Piksel')
ax.set_ylabel('Probabilitas')

fig.tight_layout()
plt.show()
"""
h, w = th.shape[:]
sumRows = []
pixelRow = []
for j in range(h):
        row = th[j:j+1, 0:w]
        pixel = np.count_nonzero(row == 0)
        sumRows += [pixel]
        pixelRow += [j]

topRow = []
for j in range(h):
        row = th[j:j+1, 0:w]
        topRow.append(np.sum(row))
        pixel = np.count_nonzero(row == 0)
        if pixel > 0:
            break

    # Search for last stroke
bottomRow = []
for j in range(h, -1, -1):
        row = th[j:j+1, 0:w]
        bottomRow.append(np.sum(row))
        pixel = np.count_nonzero(row == 0)
        if pixel > 0:
            break

mostLines = max(sumRows)/3
morePixel = [i for i in sumRows if i >= mostLines]

    # Define all zone lines
topZone = len(topRow)-1
topMiddleZone = sumRows.index(morePixel[0])
bottomMiddleZone = sumRows.index(morePixel[len(morePixel)-1])
bottomZone = img.shape[0] - len(bottomRow) + 1

print(*sumRows, sep='\n')
print(topZone, topMiddleZone, bottomMiddleZone, bottomZone)

plt.barh(pixelRow, sumRows, height=1)
plt.ylim(len(pixelRow),0)
plt.ylabel("Baris Citra (y)")
plt.xlabel("Jumlah Intensitas Piksel")
plt.show()