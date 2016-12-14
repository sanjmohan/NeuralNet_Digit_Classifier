# Sanjay Mohan
# Module with main process of application
# Constructs a GUI that handles drawing of images, displays text, and contains other text processor functionality

import os
import sys
sys.path.append(os.path.dirname(__file__ + "/../../.."))
print(sys.path)

from tkinter import *
import tkinter.filedialog
import numpy as np
import matplotlib.pyplot as plt
import win32api
import gzip
import pickle

from NeuralNet import mnistLoader
from NeuralNet.imageStandardizer import standardize
from NeuralNet import net


# If testmode is True, enables multiple debugging and accessory functionality such as viewing written images
# through matplotlib, printing classifications to console, and evaluating neural network accuracy.
# If testmode is False, module operates with normal user functionality.
testmode = False


class Gui:

    def __init__(self, master, rootWidth, rootHeight, network):
        """
        Sets up frames to hold canvas and text field
        Frames are centered vertically except buffer frame on top which spans all x
        Horiz. (rw=rootWidth): (.05rw pad)(.36rw canvas)(.05rw pad)(.01rw box)(.05rw pad)(.43rw txt)(.05rw pad)
        :param master: root panel of these frames
        :param rootWidth: width of root panel in px
        :param rootHeight: height of root panel in px
        :param network: NeuralNet.net to classify drawn digits
        """
        self.master = master
        self.screenWidth = self.master.winfo_screenwidth()
        self.screenHeight = self.master.winfo_screenheight()
        self.canvasWidth = 0.37*rootWidth  # canvas is square, height=width
        self.bufferFrame = Frame(master, width=rootWidth, height=0.05*rootHeight)
        self.canvas = Canvas(master, width=self.canvasWidth, height=self.canvasWidth, bg="white")
        self.textFrame = Frame(master, width=0.43*rootWidth, height=0.9*rootHeight, bg="white")

        # Padding applies to both sides of frame with one call
        self.bufferFrame.grid(row=0, column=0, columnspan=3)
        self.canvas.grid(row=1, column=0, padx=0.05*rootWidth)
        self.textFrame.grid(row=1, column=2, padx=0.05*rootWidth)

        # Actual text field where text appears; editable in gui
        self.textField = Text(self.textFrame, width=int(0.02*rootWidth), height=int(0.018*rootHeight), bg="white", \
                font=("Times", "24"))
        self.textField.pack(side=LEFT)

        # Scrollbar to control text field
        scroll = Scrollbar(self.textFrame, command=self.textField.yview)
        self.textField.configure(yscrollcommand=scroll.set)
        scroll.pack(side=RIGHT, fill=Y)

        self.makeMenus()
        self.bindEvents()
        self.network = network
        self.drawnPoints = np.zeros((self.screenHeight, self.screenWidth))  # holds drawn points!
        self.f = None  # for saving text
        self.drawmode = False
        self.resetTime = 1000  # time after last mouse movement before drawn image is processed
        self.lastx = -1
        self.lasty = -1
        # For generating new image sets:
        if testmode:
            self.myimages = []
            self.numberid = 0

    def bindEvents(self):
        # <B1-Motion> handles mouse movement while left clicked
        # <Motion> handles all mouse movement
        # <Button-1> handles single left mouse clicks
        # <Button-2> handles single middle mouse clicks
        # <Button-3> handles single right mouse clicks
        self.master.bind("<Motion>", self.motion)
        self.master.bind("<Button-1>", self.leftClick)
        self.master.bind("<Button-3>", self.rightClick)

    def makeMenus(self):
        # Constructs file, edit, and reset speed menus
        menu = Menu(self.master)
        self.master.config(menu=menu)

        fileMenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="New", command=self.menuNew)
        fileMenu.add_command(label="Open", command=self.menuOpen)
        fileMenu.add_command(label="Save", command=self.menuSave)
        fileMenu.add_command(label="Save As...", command=self.menuSaveAs)

        editMenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Edit", menu=editMenu)
        editMenu.add_command(label="Copy", command=self.menuCopy)
        editMenu.add_command(label="Cut", command=self.menuCut)
        editMenu.add_command(label="Paste", command=self.menuPaste)

        speedMenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Reset Speed", menu=speedMenu)
        speedMenu.add_command(label="Slow", command=self.menuSlow)
        speedMenu.add_command(label="Normal", command=self.menuNormal)
        speedMenu.add_command(label="Fast", command=self.menuFast)

    def leftClick(self, event=None):
        # Spacebar!
        self.updateText(" ")

    def rightClick(self, event=None):
        # Toggle drawing mode
        self.drawmode = not self.drawmode
        if self.drawmode:
            win32api.SetCursorPos((920, 400))
            self.lastx = -1
            self.lasty = -1

    def updateText(self, txt):
        self.textField.insert(END, txt)

    def motion(self, event):
        # Capture mouse motion, record points visually and in self.drawnPoints
        if not self.drawmode:
            return
        # Reset timer
        try:
            # If this is first time motion is called, the timer will not exist
            self.master.after_cancel(self.drawTimer)
        except AttributeError:
            pass
        self.draw(event)
        self.record(event)
        self.drawTimer = self.master.after(self.resetTime, self.inputEnd)  # similar to threading.Timer object

    def draw(self, event):
        if self.lastx == -1:
            self.lastx = event.x_root
            self.lasty = event.y_root
        # Each point appears as wide line on the canvas but is recorded as single point
        radius = self.canvasWidth / 50
        # Scale from screen size to canvas size
        xPos = int(event.x_root * self.canvasWidth / self.screenWidth)
        yPos = int(event.y_root * self.canvasWidth / self.screenHeight)
        lastxPos = int(self.lastx * self.canvasWidth / self.screenWidth)
        lastyPos = int(self.lasty * self.canvasWidth / self.screenHeight)
        self.canvas.create_line(lastxPos, lastyPos, xPos, yPos, width=radius, fill="black")

    def record(self, event):
        # Fill in every pixel between last coords and new coords;
        # sometimes quick movement causes choppy point registration; this fills in gaps
        dx = event.x_root - self.lastx
        dy = event.y_root - self.lasty
        maxDist = int(max(abs(dx), abs(dy)))
        for pt in range(maxDist):
            x = int(pt / maxDist * dx + self.lastx)
            y = int(pt / maxDist * dy + self.lasty)
            # python - where compound inequalities exist!
            if 0 <= x < self.screenWidth \
                    and 0 <= y < self.screenHeight:
                self.drawnPoints[y, x] = 0.98
        self.lastx = event.x_root
        self.lasty = event.y_root

    def inputEnd(self):
        # Called after user has input a hand drawn digit
        wasDrawModeOnBefore = self.drawmode
        self.drawmode = False
        if np.count_nonzero(self.drawnPoints) > 1:
            # To prevent processing random erroneous single points when user does not input motion
            self.identify()
        self.resetPoints()
        win32api.SetCursorPos((920, 400))
        self.lastx = -1
        self.lasty = -1
        self.drawmode = wasDrawModeOnBefore

    def identify(self, event=None):
        # Feeds points drawn into GUI into the GUI's neural network; updates text with classified digit
        pts = standardize(self.drawnPoints)
        if self.network:
            result = self.network.feedforward(pts)
            numResult = valueOfVector(result)
            self.updateText(str(numResult))
            if testmode:
                print(result)
                print("Number =", numResult)
                # For generating new images sets:
                # self.myimages.append((pts, int(self.numberid / 10)))
                # self.numberid += 1
                # print("next number:", int(self.numberid / 10))
        if testmode:
            displayPoints(pts)

    def resetPoints(self):
        self.canvas.delete(ALL)
        self.drawnPoints = np.zeros((self.screenHeight, self.screenWidth))

    def menuNew(self):
        self.resetPoints()
        self.textField.delete(1.0, END)
        self.master.title("Amazing Handwriting Interpreter")

    def menuOpen(self):
        self.menuNew()
        self.f = tkinter.filedialog.askopenfile(mode="r+", filetypes=[("Text files", "*.txt")])
        self.updateText(self.f.read())
        s = "Amazing Handwriting Interpreter - " + self.f.name
        self.master.title(s)

    def menuSaveAs(self):
        self.f = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".txt")
        if self.f is not None:
            self.f.write(str(self.textField.get(1.0, END)))
            self.f.close()
            s = "Amazing Handwriting Interpreter - " + self.f.name
            self.master.title(s)

    def menuSave(self):
        if self.f is None:
            self.menuSaveAs()
        self.f.write(str(self.textField.get(1.0, END)))
        self.f.close

    def menuCopy(self):
        try:
            self.master.clipboard_clear()
            self.master.clipboard_append(self.textField.get(SEL_FIRST, SEL_LAST))
        except TclError:  # nothing selected in text field
            pass

    def menuCut(self):
        try:
            self.menuCopy()
            self.textField.delete(SEL_FIRST, SEL_LAST)
        except TclError:  # nothing selected in text field
            pass

    def menuPaste(self):
        try:
            self.updateText(self.master.clipboard_get())
            self.textField.delete(SEL_FIRST, SEL_LAST)
        except TclError:  # nothing selected in text field
            pass

    def menuSlow(self):
        self.resetTime = 1000

    def menuNormal(self):
        self.resetTime = 500

    def menuFast(self):
        self.resetTime = 150


def displayPoints(points):
    """
    Displays points array as an image using matplotlib.pyplot; for testing
    :param points: np.array with size (784, 1)
    """
    imgplot = plt.imshow(points.reshape((28, 28)), interpolation="none")
    imgplot.set_cmap("Greys")
    plt.show()


def viewMNIST(trainingData, numImages):
    """
    :param trainingData: data set of images to view
    :param numImages: number of images in data set to view
    """
    for i in range(numImages):
        displayPoints(trainingData[i][0])


def valueOfVector(vector):
    """
    :param vector: (n, 1)d np.array
    :return: index of max value of vector
    """
    for i in range(len(vector)):
        if vector[i, 0] == vector.max():
            return i


def evaluate(network, testData):
    """
    :param network: NeuralNet.net to evaluate
    :param testData: data to evaluate network over
    :return: accuracy of network as float between 0 and 100
    """
    accuracy = 0.0
    length = len(testData)
    for x in testData:
        output = network.feedforward(x[0])
        if output.max() == output[x[1]]:
            accuracy += 1 / length
    accuracy *= 100
    print(accuracy, "% correct")
    return accuracy


def loadNetwork(name, trainingData=None, valiData=None):
    """
    Loads network from file, if it exists, or trains and returns new network
    :param name: name of saved network file or name to be saved as
    :param trainingData: data on which to train network
    :param valiData: data on which to continuously monitor accuracy of network
    :return: network loaded from file or newly trained network
    """
    try:
        return net.loadNetwork(name)
    except FileNotFoundError:
        print("Training network...")
    if trainingData is None:
        print("No training data input")
        raise AttributeError
    network = net.Network(np.array([784, 100, 10]))
    # gradientDescent(trainingData, number of epochs, size of minibatch, eta [ie learning rate])
    network.gradientDescent(trainingData, 30, 10, 0.1, valiData=valiData)
    accuracy = evaluate(network, loadMyImages("mytestimages3_expanded.pkl.gz"))
    name = name + "_" + str(int(accuracy*100))
    network.saveNetwork(name)
    print("Network saved")
    return network


def loadMyImages(name):
    """
    For loading any training or test set with images generated by this gui (doesnt need reformatting like mnist)
    :param name: name of dataset file
    :return: dataset
    """
    f = gzip.open(name, "rb")
    data = pickle.load(f, encoding="latin1")
    f.close()
    return data


def initRoot(width, height):
    """
    :param width: width in px of root panel
    :param height: height in px of root panel
    :return: formatted root panel
    """
    root = Tk()
    screenWidth = root.winfo_screenwidth()
    screenHeight = root.winfo_screenheight()
    rootPosX = int((screenWidth - width) / 2)
    rootPosY = int((screenHeight - height) / 3)
    geom = "%dx%d+%d+%d" % (width, height, rootPosX, rootPosY)
    root.geometry(geom)
    root.resizable(width=False, height=False)
    root.title("Amazing Handwriting Interpreter")
    return root


if True:
    expandedTrainingData, validationData, testData = mnistLoader.load(expanded=True, short=False)
    # for viewing sample images from mnist dataset for testing
    viewMNIST(expandedTrainingData, 10)

neuralnetwork = loadNetwork(name="mnist_exp_8520")

rootHeight = 1080
rootWidth = 1920
root = initRoot(width=rootWidth, height=rootHeight)
gui = Gui(root, rootWidth=rootWidth, rootHeight=rootHeight, network=neuralnetwork)
root.mainloop()  # makes root appear

# For generating new image sets:
# if testmode:
#     myimages = gui.myimages
#     if gui.numberid == 100:
#         f = gzip.open("mytrainimages3.pkl.gz", "w")
#         pickle.dump(myimages, f)
#         f.close()
#         print("saved")
