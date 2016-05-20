from __future__ import print_function
from __future__ import division

import os
import glob
from matplotlib import pyplot as plt
import pickle
import random
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

SUBSET = False
DOWNSAMPLE = 8
NUM_CLASSES = 2

WIDTH, HEIGHT = 352 // DOWNSAMPLE, 240 // DOWNSAMPLE

IMAGE_ROOT = os.path.expanduser('~/Doorbell/images')

def load_image(path):
    img = imread(path, as_grey=True)
    #img = imread(path, as_grey=False)
        
    # Normalize the image to have mean zero
    # Stretch the intensities so 5 - 95 percentile goes to [-1, 1]
    normalize = True
    if normalize:
        img = img.astype(float) / 255.0
        img = img - np.average(img)
        scale = np.max(np.abs(np.percentile(img, 5.0)), np.abs(np.percentile(img, 95.0)))
        img = img / scale
        img = np.clip(img, -1.0, 1.0)
    
        # Mask the timestamp
        img[220:234, 96:258] = 0.0
    
    #plt.imshow(img, cmap='gray')  #!!!
    #plt.show() #!!!

    if DOWNSAMPLE != 1:
        img = imresize(img, (HEIGHT, WIDTH))

    return img

def load_train(base):
    X_train = []
    y_train = []

    print('Reading training images...')
    for j in range(NUM_CLASSES):
        print('Loading folder c{}...'.format(j))
        paths = glob.glob(os.path.join(base, 'c{}'.format(j), '*.jpg'))

        if SUBSET:
            paths = paths[:100]

        for i, path in tqdm(enumerate(paths), total=len(paths)):
            img = load_image(path)
            if i == 0:
                imsave('c{}.jpg'.format(j), img)
           
            if len(img.shape) == 3:
                img = img.swapaxes(2, 0)
            else:
                img = img.swapaxes(1, 0)
	
            X_train.append(img)
            y_train.append(j)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    y_train = OneHotEncoder(n_values=NUM_CLASSES) \
        .fit_transform(y_train.reshape(-1, 1)) \
        .toarray()

    return X_train, y_train

def load_test(base):
    X_test = []
    y_test = []
    X_test_id = []

    print('Reading test images...')
    for j in range(NUM_CLASSES):
        print('Loading folder c{}...'.format(j))
        paths = glob.glob(os.path.join(base, 'c{}'.format(j), '*.jpg'))

        if SUBSET:
            paths = paths[:100]

        for i, path in tqdm(enumerate(paths), total=len(paths)):
            img_id = os.path.basename(path)
            
	    img = load_image(path)
            if i == 0:
                imsave('c{}.jpg'.format(j), img)
            
            if len(img.shape) == 3:
                img = img.swapaxes(2, 0)
            else:
                img = img.swapaxes(1, 0)

            X_test.append(img)
            y_test.append(j)
            X_test_id.append(img_id)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_id = np.array(X_test_id)
    return X_test, y_test, X_test_id

X_train, y_train = load_train(os.path.join(IMAGE_ROOT, 'train'))
X_test, y_test, X_test_ids = load_test(os.path.join(IMAGE_ROOT, 'test'))

if SUBSET:
    dest = 'data_{}_subset.pkl'.format(DOWNSAMPLE)
else:
    dest = 'data_{}.pkl'.format(DOWNSAMPLE)

with open(dest, 'wb') as f:
    pickle.dump((X_train, y_train, X_test, y_test, X_test_ids), f)
