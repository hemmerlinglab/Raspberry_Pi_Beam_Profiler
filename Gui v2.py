from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication, QLabel, QFileDialog, QGridLayout, QHBoxLayout, QVBoxLayout)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np
import scipy
from scipy import optimize
from PIL import Image
import sys
import os

#pixels on the rasberry pi (V.2) camera are 1.12 x 1.12 micrometers
#the pi camera has dimensions of 3280 x 2464 pixels


#array handling functions

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


#image handling functions

def get_img(file): #converts an image from the home folder to a normalized grayscale image (array)
    x = np.array(Image.open(file).convert("L"))
    x = x/255.0
    return(x)
        
def img_center(img): #gives the pixel analogous to the image's center of mass ***
    img = proj_norm(img)
    (x, y) = (img[0], img[1])
    xC = np.sum(x*range(0,len(x)))/np.sum(x)
    yC = np.sum(y*range(0,len(y)))/np.sum(y)
    return(np.round(xC,0), np.round(yC,0))
        
def img_crop(img): #crops the image around the image's center
    mid = img_center(img)
    x = int(mid[0]); y = int(mid[1])
    #finds the distance from the image's center to the edges
    edges = (mid[0], len(img[1])-mid[0], mid[1], len(img[0])-mid[1])
    edges = np.min(edges)
    #finds an approximate x and y width for the image
    nonzeros = np.where(array_norm(img)<0.1, 0, 1)
    xRad = np.max(np.count_nonzero(nonzeros, axis=0))
    yRad = np.max(np.count_nonzero(nonzeros, axis=1))
    radius = np.max((xRad,yRad))*3//4
    #compares the image's radius and the distance to the closest edge in determining a crop size
    crop = int(np.min((radius, edges)))
    return(img[y-crop:y+crop,x-crop:x+crop])

def saturation_percent(img): #gives the percentage of pixels in the beam that are saturated
    saturatedPixels = np.count_nonzero(np.where(img==1,1,0))
    totalBeamPixels = np.count_nonzero(np.where(img>=0.135,1,0))
    percent = (saturatedPixels/totalBeamPixels*100)
    return(np.round(percent,1))


#fitting functions and waist calculations

def gauss(x, a, b, c, h): # a basic form of the gaussian distribution ***
    return((np.exp(-2*np.power(((x - a)/b),2))*c)+h)

def saturated_fit(array, axis): #gives a gaussian fit, ignoring any of the saturated data points
    zeros = np.where(array==1, 1, 0)
    if axis=='x':
        zeros = np.unique(np.nonzero(zeros)[1])
        data = proj_norm(array)[0]
    if axis=='y':
        zeros = np.unique(np.nonzero(zeros)[0])
        data = proj_norm(array)[1]

    x = np.array(range(0,len(data)))
    x = np.delete(x, zeros, 0)
    y = np.delete(data, zeros, 0)
    fit = scipy.optimize.curve_fit(gauss, x, y, p0=[np.argmax(data),len(array)/4, 1.1, 0])[0]
    newFit = gauss(range(len(data)),fit[0],fit[1],fit[2],fit[3])
    return(newFit, (fit[1]*1.12), x, y)



class App(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raspberry Pi Beam Profiler")
        self.setGeometry(0,0,1200,800)
        
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)
        
        self.initUI()
        
    def initUI(self):
        #starts a dialog to open a file to process
        filePath = QFileDialog.getOpenFileName(None, 'Open File')[0]
        fileName = os.path.basename(filePath)
        
        #gets the image from the file path and resizes it
        a1 = get_img(filePath)
        a1 = img_crop(a1)
        a1 = img_crop(a1)
        
        #gets projections of image
        projection = proj_norm(a1)
        xData = projection[0]
        xIndex = np.around(1.12 * np.array(range(0,len(xData))))
        yData = projection[1]
        yIndex = np.around(1.12 * np.array(range(0,len(yData))))
        #gets fits and waist of the image
        xFit = saturated_fit(a1, 'x')
        yFit = saturated_fit(a1, 'y')
        xWaist = xFit[1]; yWaist = yFit[1]
        avgWaist = round((xWaist + yWaist)/2,1)
        
        
        #sets the parent to Qwidget to display image pixmaps, otherwise they do not display properly
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        grid = QGridLayout(self.central_widget)
        
        
        #creates a figure canvas object
        fig = FigureCanvas(Figure(figsize=(8,8)))
        beam_plot = fig.figure.subplots()
        
        #adds the beam image to the plot
        limit = round(1.12 * len(a1))
        beam_plot.imshow(a1, cmap='plasma', extent=(-1,limit,-1,limit))
        beam_plot.set_xlabel("Distance (\u03BCm)")
        beam_plot.set_ylabel("Distance (\u03BCm)")
        
        #sets up axis for the x and y plots
        divider = make_axes_locatable(beam_plot)
        xProj = divider.append_axes("top", 1.7, pad=0, sharex=beam_plot)
        xProj.xaxis.set_tick_params(labelbottom=False)
        yProj = divider.append_axes("right", 1.7, pad=0, sharey=beam_plot)
        yProj.yaxis.set_tick_params(labelleft=False)
    
        
        #creates plots for the x projection
        xProj.margins(0, 0.05)
        xProj.plot(xIndex, xFit[0], label='gaussian fit', linewidth=2.5, color='black')
        xProj.scatter(xIndex, xData, label='raw data', s=30)
        xProj.scatter(np.around(1.12 * xFit[2]), xFit[3], label='unsaturated data', s=30)
        xProj.legend(loc=1,fontsize=7)
        xProj.set_ylabel("Intensity")
        xProj.set_title("x-Projection")
        
        #creates plots for the y projection
        yProj.margins(0.05, 0)
        yProj.plot(yFit[0], yIndex, label='gaussian fit', linewidth=2.5, color='black')
        yProj.scatter(yData, yIndex, label='raw data', s=30)
        yProj.scatter(yFit[3], np.around(1.12 * yFit[2]), label='unsaturated data', s=30)
        yProj.legend(loc=1,fontsize=7)
        yProj.set_xlabel("Intensity")
        yProj.set_title("y-Projection")
        
        
        
        #creates labels to display data
        l1 = QLabel(self); l1.setText(''.join(["file name: ", str(fileName) ]))
        l2 = QLabel(self); l2.setText(''.join(["image saturation = ", str(saturation_percent(a1)),"%" ]))
        l3 = QLabel(self); l3.setText(''.join(["beam waist (x-axis) = ", str(round(xWaist,1))," \u03BCm" ]))
        l4 = QLabel(self); l4.setText(''.join(["beam waist (y-axis) = ", str(round(yWaist,1))," \u03BCm" ]))
        l5 = QLabel(self); l5.setText(''.join(["beam waist (average) = ", str(avgWaist)," \u03BCm" ]))
       
        
        #adds the labels to a sub grid
        grid2 = QGridLayout()
        grid2.addWidget(l1, 1, 0, alignment=Qt.AlignTop | Qt.AlignLeft)
        grid2.addWidget(l2, 2, 0, alignment=Qt.AlignTop | Qt.AlignLeft)
        grid2.addWidget(l3, 3, 0, alignment=Qt.AlignTop | Qt.AlignLeft)
        grid2.addWidget(l4, 4, 0, alignment=Qt.AlignTop | Qt.AlignLeft)
        grid2.addWidget(l5, 5, 0, alignment=Qt.AlignTop | Qt.AlignLeft)
        
        #starts arranging the overall layout
        grid.addWidget(fig, 0, 0, alignment=Qt.AlignTop | Qt.AlignCenter) #adds the figure to the grid
        grid.addLayout(grid2, 0, 1, alignment=Qt.AlignTop | Qt.AlignCenter) #adds the data to the grid
    
        self.show()
   

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
