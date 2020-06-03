"""
from __future__ import with_statement
from PIL import Image
 
im = Image.open('sample_image/graying.jpg') #relative path to file

#load the pixel info
pix = im.load()

#get a tuple of the x and y dimensions of the image
width, height = im.size
 
#open a file to write the pixel data
with open('output_file.csv', 'w+') as f:
  for x in range(width):
    for y in range(height):
      r = pix[y,x][0]
      g = pix[y,x][1]
      b = pix[y,x][2]
      f.write('{0} {1} {2},'.format(r,g,b))
    f.write('\n')
"""