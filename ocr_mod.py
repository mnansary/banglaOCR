#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored
from tqdm import tqdm
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
import argparse
import regex
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pytesseract
import tensorflow as tf
import os
import random
from utils import *
# ---------------------------------------------------------
# globals
# ---------------------------------------------------------
IMG_DIM    =    128  # @param
NB_CHANNEL =    1    # @param
WEIGHT_PATH=    os.path.join(os.getcwd(),'weights','modifier.h5')
#---------------------------------------------------------------
# segmentation model
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
#---------------------------------------------------------------
def get_model():
    '''
        returns the trained wieghted model
    '''
    backbone= "efficientnetb7"# @param
    model = sm.Unet(backbone,
                input_shape=(IMG_DIM,IMG_DIM,NB_CHANNEL), 
                classes=NB_CHANNEL,
                encoder_weights=None)
    model.load_weights(WEIGHT_PATH)
    return model
#---------------------------------------------------------------
def invert_img(img):
    '''
        inverts an img
    '''
    img=img/255.0
    img=1-img
    img=img*255
    img=img.astype('uint8')
    return img

def thresh_img(img):
    '''
        threshold a given grascale image
    '''
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img
# ---------------------------------------------------------
def main(args):
    '''
        prediction routine
    '''
    # kernel for dilation 
    kernel=np.ones((5,5),np.uint8)
    # read greyscale
    img=cv2.imread(args.img_path,0)
    # threshold
    img=thresh_img(img)
    # inversion
    img=invert_img(img)
    # connect words
    words=cv2.dilate(img,kernel,iterations=2)
    # labeled img
    words,_ =scipy.ndimage.measurements.label(words)
    # unique words        
    word_vals=np.unique(words)
    word_vals=word_vals[1:]
    
    #--------------------------------image data extraction-------------------
    LOG_INFO('Extracting word data')
    # word wise data
    data=[]
    for component in word_vals:
        # get the word
        idx = np.where(words==component)
        # bbox
        y,h,x,w = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        # crop word
        word = img[y:h, x:w]
        # labeled graphemes
        graphemes,_ =scipy.ndimage.measurements.label(word)
        # unique grapheme
        grapheme_vals=np.unique(graphemes)
        grapheme_vals=grapheme_vals[1:]
        grapheme_data=[]
        for grapheme_val in grapheme_vals:
            # get the grapheme 
            idx = np.where(graphemes==grapheme_val)
            # bbox
            y,h,x,w = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
            # crop grapheme
            grapheme = word[y:h, x:w]
            grapheme_data.append(grapheme)
        
        data.append(grapheme_data)
    #--------------------------------image data extraction-------------------
    
    model=get_model()
    LOG_INFO('Model Loaded')
    #--------------------------------prediction ops-------------------
    total=''    
    for word in data:
        x=[]
        for grapheme in word:
            # dilate
            grapheme=cv2.dilate(grapheme,kernel,iterations=1)    
            # invert
            grapheme=invert_img(grapheme)
            # resize
            grapheme=cv2.resize(grapheme,(IMG_DIM,IMG_DIM))
            # threshold
            grapheme=thresh_img(grapheme)
            # add third dim
            grapheme=np.expand_dims(grapheme,axis=-1)
            # add batch dim
            grapheme=np.expand_dims(grapheme,axis=0)
            # append
            x.append(grapheme)
        # stack data
        x=np.vstack(x)
        # normalize
        x=x/255.0
        y=model.predict(x)
        text=''
        for grapheme in y:
            # prediction to image
            grapheme=np.squeeze(grapheme)
            grapheme=grapheme*255
            grapheme=grapheme.astype('uint8')
            # thresh
            grapheme=thresh_img(grapheme)
            # get text prediction in tesseract
            pred = pytesseract.image_to_string(grapheme, lang='ben', config='--psm 6')
            # text add for word
            text_list=regex.findall(r"[\p{Bengali}]+",pred)
            text+=text_list[0]
        total+=f'{text} '
    #--------------------------------prediction ops-------------------
    LOG_INFO(total,mcolor='yellow')
    
            
# ---------------------------------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='script to predict based on modifier model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path',
                        required=True,
                        help="The path to the folder that contains label.csv and RAW folder") 
    
    args = parser.parse_args()
    main(args)