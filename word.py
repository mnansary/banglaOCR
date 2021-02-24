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
import sys
import pandas as pd
import numpy as np 
import cv2
import os
import random
from PIL import ImageFont, Image, ImageDraw
from utils import *
# ---------------------------------------------------------
# globals
# ---------------------------------------------------------
# the font path
FONT_PATH       =  'font.ttf'
# the desired data size
DATA_DIM        =   256 
# the number of samples to take for each sample number
SAMPLE_NUM      =   100000
# the maximum number of sample to take
MAX_SAMPLES     =   8
# max size of font
FONT_SIZES=[0,0,100,70,60,50,40,35,32]

#---------------------------------------------------------------
def stripPads(arr,val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr


def getTextImage(text,_font):
    '''
        create image from text
        args:
            text    :   the text to cleate the image of
            _font   :   the specific sized font to load
        returns:
            the written text image
    '''

    WIDTH,HEIGHT=DATA_DIM,DATA_DIM
    # RGB image
    img = Image.new('RGB', (WIDTH,HEIGHT))
    # draw object
    draw = ImageDraw.Draw(img)
    # text height width
    w, h = draw.textsize(text, font=_font) 
    # drawing in the center
    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 2), text, font=_font)
    # grayscale
    img=img.convert('L')
    # array
    img=np.array(img)
    # inversion
    img=255-img
    return img

def cleanImage(img,CHAR_DIM):
    '''
        cleans and resizes the image after stripping
        args:
            img : numpy array grayscale image
        returns:
            resized clean image
    '''
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # strip
    img=stripPads(img,255)
    h,w=img.shape
    # resize to char dim
    img=cv2.resize(img,(w,CHAR_DIM))
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img


def padImage(img):
    '''
        pads an image and resizes
    '''
    # pad updown
    h,w=img.shape
    top_bottom_pad=np.ones(((DATA_DIM-h)//2,w))*255
    img=np.concatenate([top_bottom_pad,img,top_bottom_pad],axis=0)
    img=img.astype('uint8')
    img=cv2.resize(img,(DATA_DIM,DATA_DIM))
    return img    
# ---------------------------------------------------------
def main(args):
    '''
        this creates the dataset for handwritten to printed text from bengal.ai grapheme data
    '''
    DATA_DIR        =   args.data_dir
    # the labels
    LABELS_CSV      =   os.path.join(DATA_DIR,'label.csv')
    # the raw imaged
    PNG_DIR         =   os.path.join(DATA_DIR,'RAW')
    # save dir
    DATA_DIR        =   create_dir(DATA_DIR,args.iden)
    
    # check
    if not os.path.exists(PNG_DIR):
        raise ValueError("Wrong Data directory given. No RAW png folder in data path")
    try:
        df=pd.read_csv(LABELS_CSV)
    except Exception as e:
        LOG_INFO(f"Error While reading label.csv:{e}",mcolor='red')    
        sys.exit(1)

    # read dataframe 
    df=pd.read_csv(LABELS_CSV)
    df=df[['image_id','grapheme']]
    # iterate
    sample_num=int(args.sample_num)
    FONT_SIZE=FONT_SIZES[sample_num]
    LOG_INFO(f"Sample Size:{sample_num} font:{FONT_SIZE}")
    # set font for the desired sample size
    _font = ImageFont.truetype(FONT_PATH,FONT_SIZE)
    # create images
    SAVE_DIR        =   create_dir(DATA_DIR,f'{sample_num}')
    # image saving dir
    IMG_DIR         =   create_dir(SAVE_DIR,'images')
    # target saving dir
    TGT_DIR         =   create_dir(SAVE_DIR,'targets')
    
    for i in tqdm(range(SAMPLE_NUM)):
        label_text=''
        imgs=[]
        # select the data
        _df=df.sample(n=sample_num)
        # random iden
        iden=random.randint(0,SAMPLE_NUM)
        # file name
        fname=f"{iden}_{i}_{sample_num}.png"
        
        # get labels and image
        for iid,label in zip(_df.image_id.tolist(),_df.grapheme.tolist()):
            img_path=os.path.join(PNG_DIR,f"{iid}.png")
            # read
            img=cv2.imread(img_path,0)
            # clean
            img=cleanImage(img,FONT_SIZE)
            # append
            imgs.append(img)
            label_text+=label
        
        # combine
        img=np.concatenate(imgs,axis=1)
        # process image
        img=padImage(img)
        # process target
        tgt=getTextImage(label_text,_font)
        # save
        cv2.imwrite(os.path.join(IMG_DIR,fname),img)
        cv2.imwrite(os.path.join(TGT_DIR,fname),tgt)
        
# ---------------------------------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='script to create handwritten to printed text data from bengal.ai grapheme data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir',
                        required=True,
                        help="The path to the folder that contains label.csv and RAW folder") 
    parser.add_argument('--iden',
                        required=True,
                        help="The name of the folder to save data") 
    parser.add_argument('--sample_num',
                        required=True,
                        help="The number of chars to take") 
    
    args = parser.parse_args()
    main(args)