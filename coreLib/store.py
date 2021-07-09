#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------

import os
import tensorflow as tf 
from tqdm import tqdm
import numpy as np
# ---------------------------------------------------------
# globals
# ---------------------------------------------------------
# number of images to store in a tfrecord
DATA_NUM  = 1024

#---------------------------------------------------------------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def to_tfrecord(df,save_dir,r_num):
    '''	            
      Creates tfrecords from Provided Image Paths	        
      args:	        
        df              :   dataframe that contains img_path,glabel	       
        save_dir        :   location to save the tfrecords	           
        r_num           :   record number	
    '''
    # record name
    tfrecord_name='{}.tfrecord'.format(r_num)
    # path
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for idx in range(len(df)):
            image_path=df.iloc[idx,0]
            glabel  =df.iloc[idx,1]
            
            target_path=str(image_path).replace('images','targets')
            map_path   =str(image_path).replace('images','maps').replace(".png",".npy")
            #image
            with(open(image_path,'rb')) as fid:
                image_bytes=fid.read()
            # target
            with(open(target_path,'rb')) as fid:
                target_bytes=fid.read()
            # map
            _map_data=np.load(map_path)
            _map_data=_map_data.astype("int")
            
            data ={ 'image'     :_bytes_feature(image_bytes),
                    'target'    :_bytes_feature(target_bytes),
                    'map'       :_int64_feature(_map_data),
                    'glabel'    :_int64_list_feature(glabel)
            }
            # write
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)


def genTFRecords(df,mode_dir):
    '''	        
        tf record wrapper
        args:	        
            df        :   dataframe that contains "img_path" and "glabel"	        
            mode_dir  :   location to save the tfrecords	    
    '''
    for i in tqdm(range(0,len(df),DATA_NUM)):
        
        _df=df.iloc[i:i+DATA_NUM]
        # record num
        r_num=i // DATA_NUM
        # create tfrecord
        to_tfrecord(_df,mode_dir,r_num)    