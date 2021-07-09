# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd
import random
import cv2
import numpy as np
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 
from tqdm import tqdm
from .utils import stripPads,correctPadding
tqdm.pandas()
#--------------------------------------------------------------------------------------------
class config:
    min_word_len=1
    max_word_len=10
#--------------------------------------------------------------------------------------------
def createData(ds,df,comps,font,height=32):
    '''
        creates handwriten word image
        args:
            ds       :       the dataset resource
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
    maps=[]        
    comps=[comp for comp in comps if comp is not None]
    
    for idx,comp in enumerate(comps):
        #----------------------
        # image
        #----------------------
        try:
            c_df=df.loc[df.label==comp]
            # select a image file
            idx=random.randint(0,len(c_df)-1)
        except Exception as e:
            print(comp)
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
        # target and maps
        #----------------------
        
        if comp not in mods:
        
            if idx<len(comps)-1 and comps[idx+1] in mods:
                comp=comp+comps[idx+1]
        
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

    # map 
    for _img in imgs:
        map=np.zeros(_img.shape)
        h,w=map.shape
        map[int(h/4):int(3*h/4),int(w/4):int(3*w/4)]=ds.vocab.index(comp)
        maps.append(map)
    map=np.concatenate(maps,axis=1)




    
    # create word
    label="".join(comps)
    h,w=img.shape
    min_offset=1000
    max_dim=len(comps)*h+min_offset
    # draw
    image = PIL.Image.new(mode='L', size=(max_dim,max_dim))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=label, fill=255, font=font)
        
    # invert
    img=255-img
    tgt=255-tgt  


    return img,tgt,map



def single(ds,comp_type,use_dict=True,dim=(32,128)):
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
    image,target,map=createData(ds,df,comps,font,height=h)
    return correctPadding(image,dim),correctPadding(target,dim),correctPadding(map,dim,pad_val=0),comps

