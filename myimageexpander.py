# Sanjay Mohan
# For artificially expanding my own handwritten data made through the GUI
# Expand by generating rotations of each image
# There is little worth in generating translations because these and
#  future data will be positioned using the same centering algorithm

import gzip
import pickle
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt


def displayPoints(points):
    # For displaying with matplotlib.pyplot
    # points parameter is size (784, 1)
    imgplot = plt.imshow(points.reshape((28, 28)), interpolation="none")
    imgplot.set_cmap("Greys")
    plt.show()

# file = gzip.open("datasets/mytestimages3.pkl.gz", "rb")
file = gzip.open("datasets/mytrainimages3.pkl.gz", "rb")
data = pickle.load(file, encoding="latin1")
file.close()

newimages = []
for image in data:
    imgpxls = image[0]
    imgpxls2 = np.zeros((28, 28, 3))
    # displayPoints(imgpxls)
    # Resize image so it's compatible with PIL.Image
    for y in range(28):
        for x in range(28):
            imgpxls2[y, x, 0] = imgpxls[y*28 + x]
    # PIL.Image takes 0-255 rgb values; we'll rescale the point values back to 0-1 floats later
    imgpxls2 *= 255
    img = Image.fromarray(np.uint8(imgpxls2), "RGB")
    for angle in [-15, -7, 7, 15]:
        rotation = np.asarray(img.rotate(angle))
        newimg = np.zeros((784, 1))
        # Resize back to original imgpxls shape
        for y in range(28):
            for x in range(28):
                newimg[y*28 + x] = rotation[y, x, 0]
        newimg /= 255  # scale back to 0-1 floats
        # displayPoints(newimg)
        newimages.append((newimg, image[1]))
data += newimages
random.shuffle(data)
# file = gzip.open("datasets/mytestimages3_expanded.pkl.gz", "w")
file = gzip.open("datasets/mytrainimages3_expanded.pkl.gz", "w")
pickle.dump(data, file)
file.close()
