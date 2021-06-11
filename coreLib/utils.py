#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import numpy as np 
import cv2 
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
def stripPads(arr,
              val):
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
#---------------------------------------------------------------

def padImage(img,pad_loc,pad_dim):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w=img.shape
        # pad widths
        left_pad_width =(pad_dim-w)//2
        # print(left_pad_width)
        right_pad_width=pad_dim-w-left_pad_width
        # pads
        left_pad =np.zeros((h,left_pad_width))
        right_pad=np.zeros((h,right_pad_width))
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
    else:
        # shape
        h,w=img.shape
        # pad heights
        top_pad_height =(pad_dim-h)//2
        bot_pad_height=pad_dim-h-top_pad_height
        # pads
        top_pad =np.zeros((top_pad_height,w))
        bot_pad=np.zeros((bot_pad_height,w))
        # pad
        img =np.concatenate([top_pad,img,bot_pad],axis=0)
    return img.astype("uint8")    

def correctPadding(img,dim=(64,512)):
    '''
        corrects an 
    '''
    img_height,img_width=dim
    # check for pad
    h,w=img.shape
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padImage(img,pad_loc="tb",pad_dim=img_height) 
    elif w < img_width:
        # pad
        img=padImage(img,pad_loc="lr",pad_dim=img_width)
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img 