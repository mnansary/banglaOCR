# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import os 
import random
import cv2
import numpy as np
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 

from tqdm import tqdm
from glob import glob

from coreLib.utils import stripPads,correctPadding,create_dir,LOG_INFO
from coreLib.dataset import DataSet
from coreLib.store import genTFRecords
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    # dataset object
    ds=DataSet(data_path)
    # pairs
    save_path=create_dir(save_path,"finetune")
    img_dir=create_dir(save_path,"images")
    tgt_dir=create_dir(save_path,"targets")
    # records
    rec_dir  =create_dir(save_path,"tfrecords")
    # data
    df=ds.boise_state.df
    df["img_path"]=df.filename.progress_apply(lambda x: os.path.join(ds.boise_state.dir,x))

    
    font_path     =os.path.join(ds.bangla.fonts,"Bangla.ttf")
    # create font
    font=PIL.ImageFont.truetype(font_path, size=img_height)

    
    for idx in tqdm(range(len(df))):
        try:
            # extract
            img_path=df.iloc[idx,2]
            comps=df.iloc[idx,1]
            
            #-----------------
            # image
            #-----------------
            # image and label
            img=cv2.imread(img_path,0)
            # resize (heigh based)
            h,w=img.shape 
            width= int(img_height* w/h) 
            img=cv2.resize(img,(width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)

            #-----------------
            # target
            #-----------------
            # unique values
            vals=list(np.unique(img))
            # construct target    
            tgts=[]
            # grapheme wise separation
            for v,comp in zip(vals,comps):
                if v!=0:
                    idxs = np.where(img==v)
                    y_min,y_max,x_min,x_max = np.min(idxs[0]), np.max(idxs[0]), np.min(idxs[1]), np.max(idxs[1])
                    # font
                    h=y_max-y_min
                    w=x_max-x_min
                    
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
                    width= int(img_height* w/h) 
                    tgt=cv2.resize(tgt,(width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
                    tgts.append(tgt)

            tgt=np.concatenate(tgts,axis=1)
            # resize
            h,w=img.shape 
            tgt=cv2.resize(tgt,(w,h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)

            # revalue
            img[img<255]=0
            tgt=255-tgt
            # pad correction
            img=correctPadding(img,dim=(img_height,img_width))
            tgt=correctPadding(tgt,dim=(img_height,img_width))
            # save
            cv2.imwrite(os.path.join(img_dir,f"{idx}.png"),img)
            cv2.imwrite(os.path.join(tgt_dir,f"{idx}.png"),tgt)

        except Exception as e:
            LOG_INFO(e)



    # paths    
    img_paths=[img_path for img_path in glob(os.path.join(img_dir,"*.*"))]
    # tfrecords
    genTFRecords(img_paths,rec_dir)    
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("BHOCR Pre-Training Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the input boise state data folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=32,help ="height for each grapheme: default=32")
    parser.add_argument("--img_width",required=False,default=128,help ="width dimension of word images: default=128")
    args = parser.parse_args()
    main(args)
    
    
