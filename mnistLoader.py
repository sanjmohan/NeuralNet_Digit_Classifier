# Sanjay Mohan
# Module for loading MNIST data set
# Also handles data sets derived from MNIST data set including an expanded set with translations
# and a shorter, randomly sampled set
# Written with reference to Michael Nielsen's MNIST loader module

import gzip
import pickle
import numpy as np
import random


# data set file names
mnist = "datasets/mnist.pkl.gz"
expandedmnist = "datasets/expandedmnist.pkl.gz"
shortmnist = "datasets/expandedmnist_short.pkl.gz"


def loadData(expanded, short):
    """
    Load and open mnist or expandedmnist file. Not to be called outside of module
    :param expanded: if True, returns 250k set with translations
    :param short: if True, returns 20k randomly sampled set
    :return: pre-formatted data of data set as tuple
    """
    if expanded:
        print("Retrieving expanded mnist set")
        file = getExpandedSet()
    elif short:
        print("Retrieving short mnist set")
        file = getShortSet()
    else:
        print("Retrieving standard mnist set")
        file = gzip.open(mnist, "rb")
    training, validation, test = pickle.load(file, encoding="latin1")
    file.close()
    # python - where you can return 3 things at once!
    return training, validation, test


def load(expanded=False, short=False):
    """
    Reformat arrays of images and labels so each index of the final list contains an image and corresponding label.
    To be called outside of module.
    :param expanded: if True, returns 250k set with translations
    :param short: if True, returns 20k randomly sampled set
    :return: training data, validation data, test data as tuple
    """
    training, validation, test = loadData(expanded, short)
    # For each image in training (list of pixels), reshape it into a 784-d array
    trainingImages = [np.reshape(array, (784, 1)) for array in training[0]]
    trainingLabels = [vectorize(digit) for digit in training[1]]
    # Zip the two sets (images, labels) together into a tuple
    trainingData = zip(trainingImages, trainingLabels)

    validationImages = [np.reshape(array, (784, 1)) for array in validation[0]]
    # Non-vectorized labels...we'll find out later why; zip images and labels
    validationData = zip(validationImages, validation[1])

    testImages = [np.reshape(array, (784, 1)) for array in test[0]]
    testData = zip(testImages, test[1])

    print("Loaded data set")
    return list(trainingData), list(validationData), list(testData)


def vectorize(digit):
    """
    :param digit: digit (0-9) to computer unit vector
    :return: 10-d unit vector with value 1.0 in digit location
    """
    vector = np.zeros((10, 1))
    vector[digit] = 1.0
    return vector


def getExpandedSet():
    """
    :return: Load and return or create and return expandedmnist.pkl.gz
    """
    try:
        return gzip.open(expandedmnist, "rb")
    except FileNotFoundError:
        training, validation, test = loadData(expanded=False, short=False)
        createExpandedSet(training, validation, test)
    return gzip.open(expandedmnist, "rb")


def getShortSet():
    """
    :return: Load and return or create and return expandedmnist_short.pkl.gz
    """
    try:
        return gzip.open(shortmnist, "rb")
    except FileNotFoundError:
        training, validation, test = loadData(expanded=True, short=False)
        createShortSet(training, validation, test)
    return gzip.open(shortmnist, "rb")


def createExpandedSet(training, validation, test):
    """
    Saves data set with expanded training set by translating each image n pixels in each direction
    :param training: MNIST training data to translate and save
    :param validation: MNIST validation data to save into new file
    :param test: MNIST test data to save into new file
    """
    # expanded training list is 5 times as big as original
    # pixels to be translated by - should be < 5 to prevent loss of data at borders of images
    n = 2
    newImages = []
    length = len(training[0])
    print("Creating expanded training set")
    for i in range(length):
        if i % (length / 100) == 0:
            print(i / (length / 100), "% completed expanding")
        image = np.reshape(training[0][i], (784, 1))
        translated1 = np.zeros(784)
        translated2 = np.zeros(784)
        translated3 = np.zeros(784)
        translated4 = np.zeros(784)
        for j in range(784):
            if j % 28 > n:
                translated1[j - n] = image[j]  # shift left (lower x)
            if j % 28 < 28 - n:
                translated2[j + n] = image[j]  # shift right (higher x)
            if j / 28 > n:
                translated3[j - (28 * n)] = image[j]  # shift up (lower y)
            if j / 28 < 28 - n:
                translated4[j + (28 * n)] = image[j]  # shift down (higher y)
        value = training[1][i]
        newImages.append((training[0][i], value))  # add original image
        newImages.append((translated1, value))
        newImages.append((translated2, value))
        newImages.append((translated3, value))
        newImages.append((translated4, value))
    random.shuffle(newImages)
    # zip(*...) is essentially inverse zip() function
    newTraining = list(zip(*newImages))
    file = gzip.open(expandedmnist, "w")
    pickle.dump((newTraining, validation, test), file)
    file.close()
    print("Completed expanding")


def createShortSet(training, validation, test):
    """
    Saves a new data set that contains a random 20k sample of training images from expanded set instead of 250k
    :param training: MNIST expanded training data to randomly sample and save
    :param validation: MNIST validation data to save into new file
    :param test: MNIST test data to save into new file
    """
    # Equal quantities of each digit
    trainingData = list(zip(training[0], training[1]))
    random.shuffle(trainingData)
    newImages = []
    for value in range(10):
        count = 0
        for image in trainingData:
            if image[1] == value:
                newImages.append(image)
                count += 1
            if count >= 2000:
                break
    random.shuffle(newImages)
    newTraining = list(zip(*newImages))
    file = gzip.open(shortmnist, "w")
    pickle.dump((newTraining, validation, test), file)
    file.close()
