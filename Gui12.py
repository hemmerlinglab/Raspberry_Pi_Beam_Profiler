import sys
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
import os


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
def get_img(file): #converts an image from the home folder to a normalized grayscale image
    x = np.array(Image.open(file).convert("L"))
    x = x/np.max(x)
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
    
def center_data(array, w): #takes a slice around the image's center of width w in the x and y directions
    c = img_center(array)
    x = array[int(c[1]-w//2):int(c[1]+w//2), 0:int(np.shape(array)[1])]
    y = array[0:int(np.shape(array)[0]), int(c[0]-w//2):int(c[0]+w//2)]
    return(x,y)


#fitting functions and waist calculations
def gauss(x, a, b, c, h): #a basic form of the gaussian distribution ***
    return((np.exp(-2*np.power(((x - a)/b),2))*c)+h)

def proj_fit(array, axis): #gives a gaussian fit of slice of width w, ignoring any of the saturated data points
    zeros = np.where(array>0.9, 1, 0)
    if axis=='x':
        zeros = np.unique(np.nonzero(zeros)[1])
        data = proj_norm(array)[0]
    if axis=='y':
        zeros = np.unique(np.nonzero(zeros)[0])
        data = proj_norm(array)[1]
    
    index = np.array(range(0,len(data)))
    x = np.array(range(0,len(data)))
    x = np.delete(x, zeros, 0)
    y = np.delete(data, zeros, 0)
    fit = scipy.optimize.curve_fit(gauss, x, y, p0=[np.argmax(data),len(data)/4, 1.1, 0])[0]
    waist = round(fit[1],1)
    newFit = gauss(range(len(data)),fit[0],fit[1],fit[2],fit[3])
    return(newFit, waist, x, y, data, index)



class App(QMainWindow):
    
    def __init__(self):
        super().__init__()
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
        #sets window title
        self.setWindowTitle(''.join(["Beam Profiler: ", fileName]))
        
        #gets the image from the file path and resizes it
        a1 = get_img(filePath)
        a1 = img_crop(a1)
        a1 = img_crop(a1)
        
        #gets slices of the image centered on the beam
        xProj = proj_fit(a1, 'x')
        yProj = proj_fit(a1, 'y')
        
        #gets the fit data, fit and waist of the x projection
        xFit = xProj[0]
        xWaist = xProj[1]*1.12
        xData = xProj[4]
        xIndex = xProj[5]
        #gets the fit data, fit and waist of the y projection
        yFit = yProj[0]
        yWaist = yProj[1]*1.12
        yData = yProj[4]
        yIndex = yProj[5]
        #averages the two beam waists
        avgWaist = round((xWaist + yWaist)/2,1)
        
        #start setting up gui
        #sets the parent to Qwidget to display image pixmaps, otherwise they do not display properly
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        grid = QGridLayout(self.central_widget)
        
        #creates a figure canvas object
        fig = FigureCanvas(Figure(figsize=(8,8)))
        beam_plot = fig.figure.subplots()
        
        #adds the beam image to the plot
        beam_plot.imshow(a1, cmap='plasma', extent=(-1,len(a1),-1,len(a1)))
        #labels the axis
        beam_plot.set_xlabel("Distance (\u03BCm)")
        beam_plot.set_ylabel("Distance (\u03BCm)")
        #scales the axis by 1.12 to account for pixel size
        beam_plot.xaxis.set_major_locator(plt.MultipleLocator(1.12))
        beam_plot.yaxis.set_major_locator(plt.MultipleLocator(1.12))
        #limits the number of tick marks to reduce clutter
        beam_plot.xaxis.set_major_locator(plt.MaxNLocator(10))
        beam_plot.yaxis.set_major_locator(plt.MaxNLocator(10))
        
        
        #sets up axis for the x and y plots
        divider = make_axes_locatable(beam_plot)
        xPlot = divider.append_axes("top", 1.8, pad=0, sharex=beam_plot)
        xPlot.xaxis.set_tick_params(labelbottom=False)
        
        yPlot = divider.append_axes("right", 1.8, pad=0, sharey=beam_plot)
        yPlot.yaxis.set_tick_params(labelleft=False)
        
        #creates plot for the x projection
        xPlot.margins(0, 0.1)
        xPlot.yaxis.set_major_formatter(plt.NullFormatter())
        xPlot.yaxis.set_major_locator(plt.MaxNLocator(6))
        #plots main functions
        xPlot.plot(xIndex, xFit, label='gaussian fit', linewidth=2.5, color='black')
        xPlot.scatter(xIndex, xData, label='raw data', s=25)
        xPlot.scatter(xProj[2], xProj[3], label='fit data', s=25)
        #plots the saturation point
        xPlot.axhline(y=1, label='saturation point', linewidth=1.5, color='red')
        #sets up legend
        xPlot.legend(loc=1,fontsize=7)
        #xPlot.set_title("X Projection")
        
        #creates plot for the y projection
        yPlot.margins(0.1, 0)
        yPlot.xaxis.set_major_formatter(plt.NullFormatter())
        yPlot.xaxis.set_major_locator(plt.MaxNLocator(6))
        #plots main functions
        yPlot.plot(yFit, yIndex, label='gaussian fit', linewidth=2.5, color='black')
        yPlot.scatter(yData, yIndex, label='raw data', s=25)
        yPlot.scatter(yProj[3], yProj[2], label='fit data', s=25)
        #plots the saturation point
        yPlot.axvline(x=1, label='saturation point', linewidth=1.5, color='red')
        #sets up legend
        yPlot.legend(loc=1,fontsize=7)
        #yPlot.set_title("Y Projection")
        
        
        #creates labels to display data
        l1 = QLabel(self); l1.setText(''.join(["File Name: ", str(fileName)]))
        l2 = QLabel(self); l2.setText(''.join(["Image Saturation = ", str(saturation_percent(a1)),"%" ]))
        l3 = QLabel(self); l3.setText(''.join(["Beam Waist (x-axis) = ", str(round(xWaist,1))," \u03BCm" ]))
        l4 = QLabel(self); l4.setText(''.join(["Beam Waist (y-axis) = ", str(round(yWaist,1))," \u03BCm" ]))
        l5 = QLabel(self); l5.setText(''.join(["Beam Waist (average) = ", str(avgWaist)," \u03BCm" ]))
        
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
