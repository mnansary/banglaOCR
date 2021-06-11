# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
from types import LambdaType
import pandas as pd
import random
import cv2
import numpy as np
import math
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 

from tqdm import tqdm
from glob import glob
from .utils import stripPads,correctPadding
tqdm.pandas()
#--------------------------------------------------------------------------------------------
class config:
    min_word_len=1
    max_word_len=10
#--------------------------------------------------------------------------------------------
def createData(df,comps,font,height=64):
    '''
        creates handwriten word image
        args:
            df       :       the dataframe that holds the file name and label
            font_path:       the path of the font to use   
            comps    :       the list of components 
        returns:
            img     :       word image
    '''
    
    comps=[str(comp) for comp in comps]
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    while comps[0] in mods:
        comps=comps[1:]
    # construct labels
    imgs=[]
    tgts=[]
    for comp in comps:
        #----------------------
        # image
        #----------------------
        c_df=df.loc[df.label==comp]
        # select a image file
        idx=random.randint(0,len(c_df)-1)
        img_path=c_df.iloc[idx,2] 
        # read image
        img=cv2.imread(img_path,0)
        # resize
        h,w=img.shape 
        width= int(height* w/h) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # invert
        img=255-img
        img[img>0]=255
        imgs.append(img)
        
        #----------------------
        # target
        #----------------------
        # shape    
        h,w=img.shape 
        
        min_offset=100
        max_dim=h+w+min_offset
        # draw
        image = PIL.Image.new(mode='L', size=(max_dim,max_dim))
        draw = PIL.ImageDraw.Draw(image)
        draw.text(xy=(0, 0), text=comp, fill=255, font=font)
        # create target
        tgt=np.array(image)
        tgt=stripPads(tgt,0)
        # resize
        tgt=cv2.resize(tgt,(w,h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        tgts.append(tgt)
    

    img=np.concatenate(imgs,axis=1)
    tgt=np.concatenate(tgts,axis=1)
    # create word
    label="".join(comps)
    h,w=img.shape
    min_offset=1000
    max_dim=len(comps)*h+min_offset
    # draw
    image = PIL.Image.new(mode='L', size=(max_dim,max_dim))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=label, fill=255, font=font)
    # word
    word=np.array(image)
    word=stripPads(word,0)
        
    return img,tgt,word



def single(ds,comp_type,use_dict=True,dim=(64,512)):
    '''
        creates a word image-target pair
        args:
            comp_type               :       grapheme/number/mixed
            ds                      :       the dataset object
            use_dict                :       use a dictionary word (if not used then random data is generated)
            dim                     :       dimension of the image to create
    '''
    h,_=dim

    dict_df  =ds.bangla.dictionary 

    g_df     =ds.bangla.graphemes.df 

    n_df     =ds.bangla.numbers.df 

    font_path     =os.path.join(ds.bangla.fonts,"Bangla.ttf")
    # create font
    font=PIL.ImageFont.truetype(font_path, size=h)
    
    # component selection 
    if comp_type=="grapheme":
        # dictionary
        if use_dict:
            # select index from the dict
            idx=random.randint(0,len(dict_df)-1)
            comps=dict_df.iloc[idx,1]
        else:
            # construct random word with grapheme
            comps=[]
            len_word=random.randint(config.min_word_len,config.max_word_len)
            for _ in range(len_word):
                idx=random.randint(0,len(g_df)-1)
                comps.append(g_df.iloc[idx,1])
        df=g_df
    elif comp_type=="number":
        comps=[]
        len_word=random.randint(config.min_word_len,config.max_word_len)
        for _ in range(len_word):
            idx=random.randint(0,len(n_df)-1)
            comps.append(n_df.iloc[idx,1])
        df=n_df
    else:
        df=pd.concat([g_df,n_df],ignore_index=True)
        comps=[]
        len_word=random.randint(config.min_word_len,config.max_word_len)
        for _ in range(len_word):
            idx=random.randint(0,len(df)-1)
            comps.append(df.iloc[idx,1])

    # data
    image,target,word =createData(df,comps,font,height=h)
    label="".join(comps) 
    return correctPadding(image,dim),correctPadding(target,dim),correctPadding(word,dim),label

