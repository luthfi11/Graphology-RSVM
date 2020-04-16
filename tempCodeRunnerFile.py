
    inverted = image
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - image[x][y]

    cv2.imshow('Inversi', inverted)
