import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom

"""
Things to standardize:

- size (move boundaries?) (image has pts from x = y = 4 to x = y = 25)
- position (center of mass OR ->center of boundaries<- )
- color (around each 1, make 0.5s and maybe 0.3s (2 pixels away?) (but don't override other 1s!)

note - for (784, 1) np.array: index % 28 => x-coord, and index / 28 => y-coord (y0x0 y0x1 y0x2 ... y1x0 y1x1 y1x2 y1x3)
"""

# KEEP FOR PRESENTATION
"""
# point in each adjacent cardinal direction gets half value
            self.drawnPoints[yPos * 28 + xPos + 1] = \
            self.drawnPoints[yPos * 28 + xPos - 1] = \
            self.drawnPoints[(yPos-1) * 28 + xPos] = \
            self.drawnPoints[(yPos+1) * 28 + xPos] = 0.5
"""


# img parameter must be (784, 1) np.array
def standardize(img):
    lowestX, lowestY, highestX, highestY = standardizeColor(img)
    img2, highestX, highestY = standardizePosition(img, lowestX, lowestY, highestX, highestY)
    img3 = standardizeSize(img2, lowestX, lowestY, highestX, highestY)
    return img3


# color - add gray border (half value) with radius 1 in cardinal directions
# return values of lowest and highest x-coords and y-coords of any nonzero value
def standardizeColor(img):
    lowestX = 28
    lowestY = 28
    highestX = 0
    highestY = 0
    for i in range(784):
        val = img[i]
        if val != 0.98:
            continue  # move to next in loop (ie next value of i)
        if (i+1) % 28 != 1 and img[i-1] == 0:  # if not on left edge and point to left == 0; (i+1) because i starts at 0
            img[i-1] = 0.4
        if (i+1) % 28 != 0 and i < 783 and img[i+1] == 0:  # if not on right edge and point to right == 0
            img[i+1] = 0.4
        if i > 28 and img[i-28] == 0:  # if not on top edge and point above == 0
            img[i-28] = 0.4
        if i < 756 and img[i+28] == 0:  # if not on bottom edge and point below == 0
            img[i+28] = 0.4

        # find leftmost and uppermost point for position standardization
        pos = i % 28
        if pos < lowestX:
            if pos == 0:
               lowestX = 0
            else:  # to account for gray border we just added
                lowestX = pos - 1
        elif pos > highestX:
            if pos == 28:
                highestX == 28
            else:  # to account for gray border we just added
                highestX = pos + 1

        pos = int(i / 28)
        if pos < lowestY:
            if pos == 0:
                lowestY = pos
            else:  # to account for gray border we just added
                lowestY = pos - 1
        elif pos > highestY:
            if pos == 28:
                highestY == 28
            else:  # to account for gray border we just added
                highestY = pos + 1
    return lowestX, lowestY, highestX, highestY


# KEEP FOR PRESENTATION
"""
# position - move leftmost point to x = 4, uppermost point to y = 4
# img should be within (21, 21) rectangular coords
def standardizePosition(img, lowestX, lowestY):
    # move deltaX: new_pos = old_pos + deltaX
    # move deltaY: new_pos = old_pos + 28*deltaY
    deltaX = lowestX - 4
    deltaY = lowestY - 4
    # shift x values
    for y in range(28):
        for x in range(4, 25 - deltaX):
            if deltaX > 0:  # need to shift image "left" in rectangular coords
                pos = y * 28 + x  # start from "left" (low x)
            elif deltaX < 0:  # need to shift image "right" in rectangular coords
                pos = (y+1) * 28 - x  # start from "right" (high x)
            img[pos] = img[pos + deltaX]
    # shift y values
    for x in range(22):
        for y in range(4, 25 - deltaY):
            if deltaY > 0:  # need to shift image "up" in rectangular coords
                pos = y * 28 + x  # start from "top" (low y)
            elif deltaY < 0:  # need to shift image "down" in rectangular coords
                pos = (28-y) * 28 + x  # start from "bottom" (high y)
            img[pos] = img[pos + deltaY * 28]
"""
# KEEP FOR PRESENTATION


# position - move leftmost point to x = 4, uppermost point to y = 4
# returns image with shifted position; also returns updated highestX and highestY coords of nonzero values
# img should be within (21, 21) rectangular coords
def standardizePosition(img, lowestX, lowestY, highestX, highestY):
    # move shiftX: new_pos = old_pos + shiftX
    # move shiftY: new_pos = old_pos + 28*shiftY
    shiftX = lowestX - 4
    shiftY = lowestY - 4
    img2 = np.zeros((784, 1))
    for y in range(4, 24 - shiftY):
        for x in range(4, 24 - shiftX):
            img2[(y * 28 + x)] = img[((y + shiftY) * 28 + x + shiftX)]
    return img2, highestX - shiftX, highestY - shiftY


def standardizeSize(img, lowestX, lowestY, highestX, highestY):
    # we want the range of nonzero x's and y's to be 21; this probably means enlarging original img
    # if old size is too small, increase spacing between pts and fill in holes with avg value
    # old size won't be too big because coords are scaled to 21x21 when written in gui
    # the smallest point of the final should be same as original

    # determine highest individual coordinate of nonzero point
    if highestX <= 4 and highestY <= 4:
        highest = 5
    elif highestX > highestY:
        highest = highestX
    else:
        highest = highestY

    # create matrix containing values of original image, eliminating all-zero border rows and cols
    # size of new matrix is (highest-4, highest-4), dims are furthest vertical or horizontal distance between any
    # coords ofa point
    img2 = np.zeros((highest - 4, highest - 4))
    for y in range(4, highest):
        for x in range(4, highest):
            oldPos = y * 28 + x
            img2[y - 4, x - 4] = img[oldPos]

    scaleFactor = 21 / len(img2)
    if scaleFactor > 3:
        scaleFactor = 3
    img2 = zoom(img2, scaleFactor, order=0)

    img3 = np.zeros((784, 1))
    last = len(img2) + 4  # range of points to copy over to img3 depends on size of img2
    for y in range(4, last):
        for x in range(4, last):
            newPos = y * 28 + x
            img3[newPos] = img2[y - 4, x - 4]
    return img3
