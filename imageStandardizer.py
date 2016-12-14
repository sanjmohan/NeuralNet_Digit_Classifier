# Sanjay Mohan
# Standardization algorithms for images input through the GUI
# Centers, resizes, and adds grey border to each image
# Takes image of any size, returns a (784, 1) np.array representing a 28x28 px image

import numpy as np

"""
Things to standardize:

- size (move boundaries) (image has pts from x = y = 4 to x = y = 25)
- position (center of boundaries)
- color (around each 0.98, make ~half value) (but don't override other 0.98s)

note - for (784, 1) np.array: index % 28 => x-coord, and index / 28 => y-coord (y0x0 y0x1 y0x2 ... y1x0 y1x1 y1x2 etc.)
"""


def standardize(img):
    """
    :param img: 2d np.array with dimensions of drawing area
    :return: (784, 1) np.array representing standardized image
    """
    img2 = shrink(img)
    img3 = makeBorder(img2)
    return img3


def shrink(img):
    """
    :param img: 2d np.array representing preprocessed image
    :return: (784, 1)d np.array representing smaller, centered image of parameter
    """
    lowestX, lowestY, highestX, highestY = findExtrema(img)
    maxY = len(img)
    maxX = len(img[0])
    # If the extrema were not changed from default values, there are no nonzero values in img
    if lowestX == maxX and lowestY == maxY and highestX == -1 and highestY == -1:
        return np.zeros((784, 1))
    # max() to avoid later divide-by-zero (eg if the image is a straight line, one dimension would otherwise be 0)
    imgWidth = max(highestX - lowestX, 1)
    imgHeight = max(highestY - lowestY, 1)
    # Divide by largest side of bounding box, multiply by 20 to get image with 20x20 bounding square
    # The image (ie nonzero pixel values) is 20x20 but will be recorded in a 28x28 image shaped as (784, 1) vector
    scaleFactor = 20 / max(imgHeight, imgWidth)
    img2 = np.zeros((784, 1))
    for y in range(maxY):
        for x in range(maxX):
            val = img[y, x]
            if val != 0:
                # Scales down by mutiplying position relative to center axes of image by scaleFactor determined above
                newYPos = (y - lowestY - imgHeight / 2) * scaleFactor
                newXPos = (x - lowestX - imgWidth / 2) * scaleFactor
                # Center the image
                newYPos = int(newYPos + 14)
                newXPos = int(newXPos + 14)
                img2[newYPos * 28 + newXPos] = val
    return img2


def findExtrema(img):
    """
    Finds highest and lowest coords of nonzero points of param img
    :param img: 2d np.array
    :return: lowest x-coord, lowest y-coord, highest x-coord, highest y-coord of all nonzero indices
    """
    maxY = len(img)
    maxX = len(img[0])
    lowestX = maxX
    lowestY = maxY
    highestX = highestY = -1
    for y in range(maxY):
        for x in range(maxX):
            if img[y, x] > 0:
                if y < lowestY:
                    lowestY = y
                if x < lowestX:
                    lowestX = x
                if y > highestY:
                    highestY = y
                if x > highestX:
                    highestX = x
    return lowestX, lowestY, highestX, highestY


def makeBorder(img):
    """
    Add gray border (half value) with radius 1 in cardinal directions
    :param img: np.array((784, 1))
    :return: np.array((784, 1)) with new shiny grey border!
    """
    img2 = np.zeros((784, 1))
    greyValue = 0.4
    for i in range(784):
        val = img[i]
        # move to next in loop (ie next value of i) if the point is not an original filled in point
        if val != 0.98:
            continue
        img2[i] = val
        # if not on left edge and point to left == 0; (i+1) because i starts at 0
        if (i+1) % 28 != 1 and img[i-1] == 0:
            img2[i-1] = greyValue
        # if not on right edge and point to right == 0
        if (i+1) % 28 != 0 and i < 783 and img[i+1] == 0:
            img2[i+1] = greyValue
        # if not on top edge and point above == 0
        if i > 28 and img[i-28] == 0:
            img2[i-28] = greyValue
        # if not on bottom edge and point below == 0
        if i < 756 and img[i+28] == 0:
            img2[i+28] = greyValue
    return img2
