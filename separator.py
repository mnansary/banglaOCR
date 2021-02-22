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
from utils import *
# ---------------------------------------------------------
# globals
# ---------------------------------------------------------
# final image dimension
IMG_DIM=256
# the desired data size
DATA_HEIGHT     =   64
# the number of samples to take for each sample number
SAMPLE_NUM      =   50000
# the maximum number of sample to take
MAX_SAMPLES     =   8
# color maps
COLORS=[(255,0,0), # red
        (0,255,0), # green
        (0,0,255), # blue
        (255,255,0), # col-4
        (255,0,255), # col-5
        (0,255,255), # col-6
        (255,255,255),# col-7
        (128,128,128) # col-8
       ]
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

def cleanImage(img):
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
    # get shape
    h,w=img.shape
    WIDTH=int(DATA_HEIGHT*(w/h))
    # resize to char dim
    img=cv2.resize(img,(WIDTH,DATA_HEIGHT))
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def padImage(img):
    '''
        pads an image and resizes
    '''
    # pad updown
    h,w,d=img.shape
    top_bottom_pad=np.ones(((IMG_DIM-h)//2,w,d))*255
    img=np.concatenate([top_bottom_pad,img,top_bottom_pad],axis=0)
    img=img.astype('uint8')
    img=cv2.resize(img,(IMG_DIM,IMG_DIM))
    return img

def padMask(img):
    '''
        pads a mask and resizes 
    '''
    # pad updown
    h,w,d=img.shape
    top_bottom_pad=np.zeros(((IMG_DIM-h)//2,w,d))
    img=np.concatenate([top_bottom_pad,img,top_bottom_pad],axis=0)
    img=img.astype('uint8')
    img=cv2.resize(img,(IMG_DIM,IMG_DIM))
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
    # image saving dir
    IMG_DIR         =   create_dir(DATA_DIR,'images')
    # target saving dir
    TGT_DIR         =   create_dir(DATA_DIR,'targets')
    
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
    for sample_num in range(1,MAX_SAMPLES+1):
        LOG_INFO(f"Sample Size:{sample_num}")
        # create images
        for i in tqdm(range(SAMPLE_NUM)):
            imgs=[]
            tgts=[]
            # select the data
            _df=df.sample(n=sample_num)
            # random iden
            iden=random.randint(0,SAMPLE_NUM)
            # file name
            fname=f"{iden}_{i}_{sample_num}.png"
            
            # get labels and image
            for iid,cid in zip(_df.image_id.tolist(),range(SAMPLE_NUM)):
                img_path=os.path.join(PNG_DIR,f"{iid}.png")
                # read
                img=cv2.imread(img_path,0)
                # clean
                img=cleanImage(img)
                x,y=np.where(img==0)
                img=np.expand_dims(img,axis=-1)
                img=np.concatenate([img,img,img],axis=-1)
                # tgt
                tgt=np.zeros(img.shape)
                tgt[:,:,:]=COLORS[cid]
                tgt[x,y,:]=(0,0,0)
                tgt=tgt.astype('uint8')
                # append
                imgs.append(img)
                tgts.append(tgt)
            # combine
            img=np.concatenate(imgs,axis=1)
            img=padImage(img)
            tgt=np.concatenate(tgts,axis=1)
            tgt=padMask(tgt)
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
    
    args = parser.parse_args()
    main(args)