#!/usr/bin/python3
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
from tqdm import tqdm
from glob import glob
from coreLib.dataset import DataSet
from coreLib.utils import create_dir
from coreLib.words import single
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
    nb_train    =   int(args.nb_train)
    # dataset object
    ds=DataSet(data_path)
    # pairs
    img_dir=create_dir(save_path,"images")
    tgt_dir=create_dir(save_path,"targets")
    # records
    rec_dir  =create_dir(save_path,"tfrecords")

    # create the images
    for i in tqdm(range(nb_train)):
        try:
            # selection
            comp_type =random.choices(population=["number","grapheme"],weights=[0.1,0.9],k=1)[0]
            use_dict  =random.choices(population=[True,False],weights=[0.9,0.1],k=1)[0]
            img,tgt=single(ds,comp_type,use_dict,(img_height,img_width))
            
            # save
            cv2.imwrite(os.path.join(img_dir,f"{i}.png"),img)
            cv2.imwrite(os.path.join(tgt_dir,f"{i}.png"),tgt)
            
        except Exception as e:
            print(e)

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
    parser.add_argument("data_path", help="Path of the input data folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension of word images: default=512")
    parser.add_argument("--nb_train",required=False,default=1000000,help ="number of images for training:default:1000000")
    args = parser.parse_args()
    main(args)
    
    