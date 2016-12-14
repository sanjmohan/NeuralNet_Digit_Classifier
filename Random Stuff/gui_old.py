from tkinter import *
import numpy as np
import matplotlib.pyplot as plt

from NeuralNet import mnistLoader
from NeuralNet.imageStandardizer_old import standardize
from NeuralNet import imageStandardizer

from NeuralNet import network2
from NeuralNet.network2 import Network


class DrawingSpace:

    def __init__(self, width, height, network=None):
        self.root = Tk()
        self.width = width
        self.height = height
        self.canvas = Canvas(self.root, width=self.width, height=self.height) # canvas used for pix
        self.canvas.bind("<B1-Motion>", self.leftClick)
        self.canvas.bind("<Button-3>", self.rightClick)
        self.canvas.bind("<Button-2>", self.middleClick)
        self.canvas.pack()  # make them show

        self.network = network

        self.drawnPoints = np.zeros((784, 1))
        self.drawnPoints2 = np.zeros((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))

    def leftClick(self, event):
        self.canvas.create_oval(event.x-20, event.y-20, event.x+20, event.y+20, fill='black')
        # python - where compound inequalities exist!
        if 0 <= event.x < self.width and 0 <= event.y < self.height:
            xPos = int(event.x * 21/self.width) + 4  # converts from dim. of input area to 21x21, puts it in (784, 1) vector
            yPos = int(event.y * 21/self.height) + 4
            self.drawnPoints[yPos * 28 + xPos] = 0.98
        self.drawnPoints2[event.y_root, event.x_root] = 0.98

    def rightClick(self, event):
        pts = standardize(self.drawnPoints)
        pts2 = imageStandardizer.standardize(self.drawnPoints2)

        if self.network:
            result = self.network.feedforward(pts)
            print("Number =", valueOfVector(result))

        self.displayPoints(pts)
        self.displayPoints(pts2)

    def middleClick(self, event):
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill='white')
        self.drawnPoints = np.zeros(784)

    # points parameter is size (784, 1)
    def displayPoints(self, points):
        imgplot = plt.imshow(points.reshape((28, 28)), interpolation="none")
        imgplot.set_cmap("Greys")
        plt.show()

    def mainloop(self):
        self.root.mainloop()


def viewMNIST(trainingData, numImages):
    for i in range(numImages):
        DrawingSpace.displayPoints(None, (trainingData[i][0]))


def valueOfVector(vector):
    for i in range(len(vector)):
        if vector[i, 0] == vector.max():
            return i


trainingData, validationData, testData = mnistLoader.load()
viewMNIST(trainingData, 0)

# superduper = Network(np.array([784, 100, 10]), network2.QuadraticCost)
# print("going now")
# superduper.SGD(trainingData, 30, 10, 3.0)
# superduper.save("netwerk")

def evaluate():
    superduper = network2.load("net_stan_set_97_old")
    numCorrect = 0
    for i in range(len(testData)):
        result = superduper.feedforward(testData[i][0])
        if result.max() == result[testData[i][1]]:
            numCorrect += 1
        if i % 1000 == 0:
            print(i / 1000, "/ 10 complete")
    print(numCorrect / 10000 * 100, "% correct")
    return superduper

superduper = evaluate()

ds = DrawingSpace(600, 600, superduper)
ds.mainloop()  # so it doesnt disappear immediately