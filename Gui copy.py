import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt

from scipy import ndimage
from scipy import optimize
import scipy

from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QGridLayout, QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import (QImage, QPixmap, QIcon)
from PyQt5.QtCore import (Qt, pyqtSlot)
import sys

#array handling

def array_norm(n): #normalizes an array (sets all values inbetween 0 and 1) ***
    x = n - np.min(n)
    x = n/np.max(n)
    return(x)

def proj_norm(n): #projects an array onto the x and y axis and normalizes the values ***
    xNorm = np.sum(n,0)
    xNorm = xNorm/np.amax(xNorm)
    yNorm = np.sum(n,1)
    yNorm = yNorm/np.amax(yNorm)
    return(xNorm,yNorm)

#image handling

def get_img(file): #converts an image from the home folder to a normalized grayscale image (array)
    x = np.array(Image.open(file).convert("L"))
    return(x)
        
def img_center(img): #gives the pixel analogous to the image's center of mass ***
    img = proj_norm(img)
    (x, y) = (img[0], img[1])
    xC = np.sum(x*range(0,len(x)))/np.sum(x)
    yC = np.sum(y*range(0,len(y)))/np.sum(y)
    return(np.round(xC,0), np.round(yC,0))
        
def img_focus(n, xDim, yDim): #gives an image cropped around the image's "center" with dimensions:(xDim,yDim)
    mid = img_center(n)
    x = int(mid[0]); y = int(mid[1])
    xDim = xDim//2; yDim = yDim//2
    return(n[y-yDim:y+yDim,x-xDim:x+xDim])


        
#fitting functions and waist calculations
        
def gauss(x, a, b, c): # a basic form of the gaussian distribution ***
    return((np.exp(-2*np.power(((x - a)/b),2))*c))

def saturated_fit2(img, axis): #removes any oversaturated points from an image, outputs a fit array, the waist calculation and the x and y of included data
    if axis=='x':
        data = proj_norm(img)[0]
        zeros = np.nonzero(np.where(img==1,img,0))[1]
    if axis=='y':
        data = proj_norm(img)[1]
        zeros = np.nonzero(np.where(img==1,img,0))[0]

    x = np.array(range(0,len(data)))
    x = np.delete(x, zeros, 0)
    y = np.delete(data, zeros, 0)

    fit = scipy.optimize.curve_fit(gauss, x, y, p0=[np.argmax(data),100,1])[0]
    newFit = gauss(range(len(data)),fit[0],fit[1],fit[2])
    waist = (fit[1]*1.12)
    return(newFit, waist, x, y)

def trim_array(data, floor, limit): #deletes values from a data set above the limit or below the floor
    index = range(0,len(data))
    data = np.dstack((index, data))[0]
    
    a = np.array([])
    for n in index:
        if (data[n])[1] < floor or (data[n])[1] > limit:
            a = np.append(a, n)
    a = a.astype(int)
    
    out = np.delete(data,a,0)
    x = out.flatten()[0::2]
    y = out.flatten()[1::2]
    return(x,y)

def saturated_fit(data, a, b): #performs a gaussian fit for a saturated beam ***
    trim = trim_array(data, a, b)
    x = trim[0]; y = trim[1];
    fit = scipy.optimize.curve_fit(gauss, x, y, p0=[np.argmax(data),100,1])[0]
    new = gauss(range(len(data)),fit[0],fit[1],fit[2])
    return(new, np.around(fit[1]*1.12,1))


class App(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pi Beam Profiler")
        self.initUI()
        
    def initUI(self):
        #Gui creation functions
        
        def make_button(label, pos, func): #creates a basic pushbutton
            button = QPushButton(label, self)
            button.move(pos[0],pos[1])
            button.clicked.connect(func)
        
        def display_array(n): # displays an array
            img = ImageQt(Image.fromarray(n))
            pixmap = QPixmap.fromImage(img)
            label = QLabel(self)
            label.setPixmap(pixmap)
            self.resize(pixmap.width(),pixmap.height())
        
        def img_pixmap(file): #creates a pixmap of an image
            label = QLabel(self)
            pixmap = QPixmap(imagePath)
            label.setPixmap(pixmap)
            self.resize(pixmap.width(),pixmap.height())
        
        def array_pixmap(n): #makes a pixmap from a numpy array
            img = Image.fromarray(n)
            img = ImageQt(img)
            return(QPixmap.fromImage(img))
        
        #get the image you want to process
        imagePath = QFileDialog.getOpenFileName(None, 'Open File')[0]
        a1 = get_img(imagePath)
    
        #resize the image
        a1 = img_focus(a1, 1000, 1000)
        a1 = img_focus(a1, 700, 700)
        #pixels on the rasberry pi (V.2) camera are 1.12 x 1.12 micrometers
        #the unicode value for mu is \u03BC
        
        #gets projections of image
        projection = proj_norm(a1)
        xData = projection[0]
        xIndex = np.around(1.12 * np.array(range(0,len(xData))))
        yData = projection[1]
        yIndex = np.around(1.12 * np.array(range(0,len(yData))))
        
        #gets fits and waist of the img
        xFit = saturated_fit(xData, 0.1,0.5)
        xWaist = xFit[1]
        yFit = saturated_fit(yData, 0.1, 0.5)
        yWaist = yFit[1]
        avgWaist = (xWaist + yWaist)/2
        
        
        #sets the parent to Qwidget to display image pixmaps, otherwise they do not display properly
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        grid = QGridLayout(self.central_widget)
        
        scale = 350;
        
        #makes a pixmap of the beam image
        beamPix = array_pixmap(a1)
        beamPix = beamPix.scaled(scale, scale, Qt.KeepAspectRatio)
        #adds beam pixmap to label
        beamImg = QLabel(self)
        beamImg.setPixmap(beamPix)
        
        
        #create a figure and adds the arguements
        fig = plt.figure()
        xAxis = fig.subplots()
        #adds the two plots to the figure
        xAxis.plot(xIndex, xData, label='raw data')
        xAxis.plot(xIndex, xFit[0], label='gaussian fit')
        #adds labels and a legend
        xAxis.legend()
        xAxis.set_xlabel("Distance (\u03BCm)")
        xAxis.set_ylabel("Intensity")
        xAxis.set_title("x Projection")
        #saves the figure to a file and converts to a pixmap
        fig.savefig('xAxis.png', bbox_inches='tight')
        plt.close(fig)
        #converts the figure to a pixmap and scales
        p1 = array_pixmap(np.array(Image.open('xAxis.png')))
        p1 = p1.scaled(2*scale, scale, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        #adds pixmap to a label
        l1 = QLabel(self)
        l1.setPixmap(p1)
        
        
        #create a figure and adds the arguement
        fig = plt.figure()
        yAxis = fig.subplots()
        #adds the two plots to the figure
        yAxis.plot(yIndex, yData, label='raw data')
        yAxis.plot(yIndex, yFit[0], label='gaussian fit')
        #adds labels and a legend
        yAxis.legend()
        yAxis.set_xlabel("Distance (\u03BCm)")
        yAxis.set_ylabel("Intensity")
        yAxis.set_title("y Projection")
        #saves the figure to a file and converts to a pixmap
        fig.savefig('yAxis.png', bbox_inches='tight')
        plt.close(fig)
        #converts the figure to a pixmap and scales
        p2 = array_pixmap(np.array(Image.open('yAxis.png')))
        p2 = p2.scaled(2*scale, scale, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        #adds pixmap to a label
        l2 = QLabel(self)
        l2.setPixmap(p2)
        
        l3 = QLabel(self)
        l3.setText(''.join(["beam waist = ", str(avgWaist),"\u03BCm" ]))
        
        
        
        
        #starts with the layout of the window
        grid.addWidget(beamImg, 0, 0, alignment=Qt.AlignTop | Qt.AlignLeft)
        grid.addWidget(l1, 0, scale, alignment=Qt.AlignTop | Qt.AlignLeft)
        grid.addWidget(l2, scale, scale, alignment=Qt.AlignTop | Qt.AlignLeft)
        grid.addWidget(l3, scale/2, 0, alignment=Qt.AlignTop | Qt.AlignCenter)
        self.setLayout(grid)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
