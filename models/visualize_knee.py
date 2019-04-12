import os
import sys
import pylab
import glob

import vtk
import numpy
from mayavi.mlab import *
from mayavi import mlab
import numpy as np

#We order all the directories by name
path="/home/manivasagam/code/fastMRIPrivate/visualizations/file1000314/"
tulip_files = [t for t in os.listdir(path)]
tulip_files.sort() #the os.listdir function do not give the files in the right order so we need to sort them

#Function that open all the images of a folder and save them in a images list
def imageread(filePath):
    filenames = [img for img in glob.glob(filePath)]
    filenames.sort()

    temp = pylab.imread(filenames[0])
    temp = temp[:, :, :3].mean(axis=2).squeeze()
    temp = (temp / temp.max()) * 255
    d, w = temp.shape
    h = len(filenames)

    volume = np.zeros((w, d, h), dtype=np.uint16)
    k=0
    for img in filenames:
        temp = pylab.imread(img)
        temp = temp[:, :, :3].mean(axis=2).squeeze()
        temp =(temp / temp.max()) * 255
        volume[:,:,k] = temp
        k+=1
    return volume

matrix_full = imageread(path+'*.png')
matrix_full = np.repeat(matrix_full, 5, axis=2)
print(matrix_full.shape)
mlab.contour3d(matrix_full, transparent=True)
mlab.show()