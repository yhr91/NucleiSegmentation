## Helper functions for Nuclei segmentation work
## yr897021

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from skimage.morphology import label


### ------------ Data handling ----------------
def save_images(X_train, Y_train, x_str = 'images.pickle', y_str = 'labels.pickle'):
    
    import pickle
    with open(x_str, 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(y_str, 'wb') as handle:
        pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_images(x_str = 'images.pickle', y_str = 'labels.pickle'):
    import pickle

    with open('images.pickle', 'rb') as handle:
        X_train = pickle.load(handle)

    with open('labels.pickle', 'rb') as handle:
        Y_train = pickle.load(handle)
        
    return X_train, Y_train
        

### ------------ Visualization related ----------------
def overlay(img, img_ROI):
    img_norm = img/np.max(img)
    return np.dstack([img_norm, img_norm, img_ROI])

### ------------ Preprocessing ----------------
def get_bw(img_list, map_list, col=False):
    ## Returns single channel b/w images

    imgs = []
    maps = []
    
    for idx, image in enumerate(img_list):
        
        # Compare first two channels to determine b&w
        mono = np.all(image[:,:,0] == image[:,:,1])
        
        if(mono and not col):
            imgs.append(image[:,:,0])
            maps.append(map_list[idx])
            
        elif(not mono and col):
            imgs.append(image[:,:,:3])
            maps.append(map_list[idx])
            
    return imgs, maps


def get_bw_idx(img_list, col=False):
    ## Returns single channel b/w images

    imgs = []
    idxs = []
    
    for idx, image in enumerate(img_list):
        
        # Compare first two channels to determine b&w
        mono = np.all(image[:,:,0] == image[:,:,1])
        
        if(mono and not col):
            imgs.append(image[:,:,0])
            idxs.append(idx)
            
        elif(not mono and col):
            imgs.append(image[:,:,:3])
            idxs.append(idx)
            
    return imgs, idxs

## Create the minimum number of stepxstep patches to account for the full image

# Get the X/Y range of indices over which patches can be extracted
def get_x_rng(img, step):
    x_rng = range(0, img.shape[0], step)
    if x_rng[-1] > img.shape[0]-step:
        x_rng[-1] = img.shape[0]-step
    return x_rng
        
def get_y_rng(img, step):
    y_rng = range(0, img.shape[1], step)
    if y_rng[-1] > img.shape[1]-step:
        y_rng[-1] = img.shape[1]-step
    return y_rng

# Extract patches of given size
def get_patches(img, step, one_only=False):
    
    x_rng = get_x_rng(img, step) 
    y_rng = get_y_rng(img, step)
    
    # If only using one patch - try to get as central as you can
    if one_only:
        x_rng = [x_rng[len(x_rng)/2]]
        y_rng = [y_rng[len(y_rng)/2]]

    patches = []
    for x in x_rng:
        for y in y_rng:
            patches.append(img[x:x+step,y:y+step])
            
    return np.array(patches)

def find_bfield(img_list, invert=False, thresh=150):
    idxs = []
    
    for i, img in enumerate(img_list):
        if np.mean(img) > thresh:
            idxs.append(i)
            
            if (invert):
                img_list[i] = np.invert(img)
    return idxs

### ------------ Network/Training ----------------



### ------------ Result submission ----------------

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)