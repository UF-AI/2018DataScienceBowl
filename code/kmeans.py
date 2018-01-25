# -*- coding: utf-8 -*-

"""
Created on Sat Jan 20 00:57:24 2018
@author: mason rawson
"""

import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import warnings
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images,show
from skimage.transform import resize
from skimage.morphology import label
from itertools import chain

#%%
# Set some parameters
IMG_WIDTH = 350
IMG_HEIGHT = 350
IMG_CHANNELS = 3
TRAIN_PATH = './Dropbox/projects/dataSciBowl2018/input/train/'
TEST_PATH = './Dropbox/projects/dataSciBowl2018/input/test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
#%%
# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
#%%
# Get and resize train images and masks
#images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
labels = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#images, labels

#%%
for i in range(len(train_ids)):
    id_ = train_ids[i]
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    #img = imread(path + '/images/' + id_ + '.png', as_grey=True)[:,:]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for rownum in range(IMG_WIDTH):
       for colnum in range(IMG_WIDTH):
           img[rownum,colnum,0] = np.average(img[rownum,colnum,:])
    images[i] = img[:,:,0]
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)  
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    labels[i] = mask
    
#%%
for i in range(len(train_ids)):
    grey = np.zeros([IMG_WIDTH, IMG_WIDTH]) # init 2D numpy array
    # get row number
    for rownum in range(IMG_WIDTH):
       for colnum in range(IMG_WIDTH):
          grey[rownum,colnum] = np.average(images[i,rownum,colnum,:])
    images[i,:,:,0] = grey

#%%
X_train = images[:int(0.9*len(train_ids)),:,:]
Y_train = labels[:int(0.9*len(train_ids)),:,:]
Y_train = Y_train.astype(np.float32)

X_validate = images[int(0.9*len(train_ids)):,:,:]
Y_validate = labels[int(0.9*len(train_ids)):,:,:]
Y_validate = Y_validate.astype(np.float32)
#%%
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
for i in range(len(test_ids)):
    id_ = test_ids[i]
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[i] = img
#%%

def shuffle():
    global images, labels
    p = np.random.permutation(len(X_train))    
    images = X_train[p]
    labels = Y_train[p]
def next_batch(batch_s, iters):
    if(iters == 0):
        shuffle()
    count = batch_s * iters
    return images[count:(count + batch_s)], labels[count:(count + batch_s)]

#%%
data_ = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=20)
pca.fit(img[:,:,0])
pca.explained_variance_ratio_
pca.fit_transform(img[:,:,0])
#%%
PICS = 25
BOX = IMG_WIDTH/10
IMG_CHANNELS = 1


#%%
STRIDE = BOX
avg = np.zeros([PICS,int(IMG_WIDTH/BOX),int(IMG_HEIGHT/BOX),IMG_CHANNELS])
var = np.zeros([PICS,int(IMG_WIDTH/BOX),int(IMG_HEIGHT/BOX),IMG_CHANNELS])

for im_idx in range(PICS):
#    for ch in range(IMG_CHANNELS):
    for w in range(int(IMG_WIDTH/BOX)): 
        for h in range(int(IMG_HEIGHT/BOX)):
            avg[im_idx,w,h] = np.average(X_train[im_idx,int(w*BOX):int((w+1)*BOX),int(h*BOX):int((h+1)*BOX)])
            var[im_idx,w,h] = np.var(X_train[im_idx,int(w*BOX):int((w+1)*BOX),int(h*BOX):int((h+1)*BOX)])
#%%
avg_samp = np.zeros([PICS,int((IMG_WIDTH/BOX) * (IMG_HEIGHT/BOX) * IMG_CHANNELS)])
var_samp = np.zeros([PICS,int((IMG_WIDTH/BOX) * (IMG_HEIGHT/BOX) * IMG_CHANNELS)])
for pics in range(PICS):
    avg_samp[pics,:] = avg[pics,:,:,:].flatten()
    var_samp[pics,:] = var[pics,:,:,:].flatten()
    
features = np.append(avg_samp,var_samp,axis=1)

#%%
kmeans = KMeans(n_clusters=10).fit_predict(features)

#%%
X_tf = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 3])
Y_tf = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 1])
lr = tf.placeholder(tf.float32)
#%%
def deconv2d(input_tensor, filter_size, output_size, out_channels, in_channels, name):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]    
    out_shape = tf.stack([batch_size, output_size, output_size, out_channels])    
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    strides = [1, 2, 2, 1]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='VALID')
    return h1


Y1 = tf.layers.conv2d(X_tf, filters=16, kernel_size=3, strides=1, padding="VALID", activation=tf.nn.relu)
Y2 = tf.layers.conv2d(Y1, filters=32, kernel_size=2, strides=2, padding="VALID", activation=tf.nn.relu)
Y3 = tf.layers.conv2d(Y2, filters=64, kernel_size=3, strides=2, padding="VALID", activation=tf.nn.relu)
Y4 = tf.layers.conv2d(Y3, filters=64, kernel_size=3, strides=2, padding="VALID", activation=tf.nn.relu)

Y3_ = deconv2d(Y4, 4, 32, 32, 64, "Y3_deconv")
Y3_ = tf.nn.relu(Y3_)

Y2_ = deconv2d(Y3_, 2, 64, 16, 32, "Y2_deconv")
Y2_ = tf.nn.relu(Y2_)

logits = deconv2d(Y2_, 2, 350, 1, 16, "logits_deconv")
#%%
loss = tf.reduce_mean(tf.square(Y_tf - logits))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

iter_count = 0

for i in range(12000):
    # training on batches of 5 images with 5 mask images
    if(iter_count > 120):
        iter_count = 0    

    batch_X, batch_Y = next_batch(5, iter_count)

    iter_count += 1

    feed_dict = {X_tf: batch_X, Y_tf: batch_Y, lr: 0.0001}
    loss_value = sess.run([loss], feed_dict=feed_dict)

    if(i % 500 == 0):
        print("training loss:", str(loss_value))
        
        #print("training acc:" + str(acc))
        #test_data = {X: X_validate, Y_: Y_validate}
        #test_acc, tests_loss = sess.run([accuracy, loss], feed_dict=test_data)
print("Done!")
#%%



