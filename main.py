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
import os
import random
import tensorflow as tf 

from utils import *
from glob import glob
from tqdm import tqdm
# ---------------------------------------------------------
# globals
# ---------------------------------------------------------
# number of images to store in a tfrecord
DATA_NUM        = 10240
#---------------------------------------------------------------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(image_paths,save_dir,r_num):
    '''	            
      Creates tfrecords from Provided Image Paths	        
      args:	        
        image_paths     =   specific number of image paths	       
        save_dir        =   location to save the tfrecords	           
        r_num           =   record number	
    '''
    # record name
    tfrecord_name='{}.tfrecord'.format(r_num)
    # path
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    LOG_INFO(tfrecord_path)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for image_path in tqdm(image_paths):
            target_path=str(image_path).replace('images','targets')
            #imae
            with(open(image_path,'rb')) as fid:
                image_png_bytes=fid.read()
            # target
            with(open(target_path,'rb')) as fid:
                target_png_bytes=fid.read()
            # get count
            samples=int(str(os.path.basename(image_path)).replace(".png",'').split("_")[-1])
            # data
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'target':_bytes_feature(target_png_bytes),
                    'samples':_int64_feature(samples)
            }
            # write
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)


def genTFRecords(__paths,mode_dir):
    '''	        
        tf record wrapper
        args:	        
            __paths   =   all image paths for a mode	        
            mode_dir    =   location to save the tfrecords	    
    '''
    for i in tqdm(range(0,len(__paths),DATA_NUM)):
        # paths
        image_paths= __paths[i:i+DATA_NUM]
        # record num
        r_num=i // DATA_NUM
        # create tfrecord
        to_tfrecord(image_paths,mode_dir,r_num)    

# ---------------------------------------------------------
def main(args):
    '''
        this creates the tfrecords for the images and targets
    '''
    data_dir = args.data_dir
    # path of images dir
    img_dir  = os.path.join(data_dir,"images")
    # check
    if not os.path.exists(img_dir):
        raise ValueError("Wrong directory given for images")
    
    # img paths
    img_paths=[img_path for img_path in tqdm(glob(os.path.join(img_dir,'*.*')))]
    # shiffle
    random.shuffle(img_paths)
    # test -train split
    nb_train = int(len(img_paths)*0.8)
    train_paths = img_paths[:nb_train]
    eval_paths  = img_paths[nb_train:]

    # tfrecord saving directories 
    save_dir = data_dir
    rec_dir  = create_dir(save_dir,'tfrecords')
    train_rec= create_dir(rec_dir,'train')
    eval_rec = create_dir(rec_dir,'eval')
    
    LOG_INFO(f"Train records:{train_rec}")
    LOG_INFO(f"Eval records:{eval_rec}")

    # tfrecords
    genTFRecords(train_paths,train_rec)
    genTFRecords(eval_paths,eval_rec)

    LOG_INFO(f'Train Images:{len(train_paths)} Eval Images:{len(eval_paths)}')

# ---------------------------------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='script to create handwritten to printed text data from bengal.ai grapheme data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir',
                        required=True,
                        help="The path to the folder that contains label.csv and RAW folder") 
    args = parser.parse_args()
    main(args)