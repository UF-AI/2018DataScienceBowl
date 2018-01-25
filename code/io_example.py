"""
Created on Fri Jan 19 10:42:37 2018
@author: christian marin
simple i/o example using skimage
"""

# general imports
import os
from skimage import io

### data management ###
# todo: optimize by switching from obj to np array
# where [0] = img, and [>0] = corresponding masks

class TrainSample():
    ''' training data container '''
    imgPath = ''
    maskPaths = []
    def __init__(self, imgPath, maskPaths):
        self.imgPath = imgPath
        self.maskPaths = maskPaths

class TestSample():
    ''' testing data container '''
    imgPath = ''
    def __init__(self, imgPath):
        self.imgPath = imgPath

def indexTrainData(root):
    ''' loads training data from data directory'''
    data = []
    for d in os.listdir(path=root):
        imgPath = root+'/'+d+'/images/'+os.listdir(root+'/'+d+'/images/')[0]
        maskPaths = os.listdir(root+'/'+d+'/masks/')
        (root+'/'+d+'/masks/'+m for m in maskPaths)
        data.append(TrainSample(imgPath,maskPaths))
    return data

def indexTestData(root):
    ''' loads testing data from data directory'''
    data = []
    for d in os.listdir(path=root):
        imgPath = root+'/'+d+'/images/'+os.listdir(root+'/'+d+'/images/')
        data.append(TrainSample(imgPath))
    return data


### test i/o ###

if __name__ == "__main__":
    trainDir = '../data/stage1_train/'
    testDir = '../data/stage1_test/'
    trainData = indexTrainData(trainDir)
    
    image = io.imread(trainData[3].imgPath)[:,:,0:3]
    io.imshow(image)
    io.show()
