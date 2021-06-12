#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import tensorflow as tf
import easyocr
import pytesseract
import scipy
import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

from glob import glob
from tqdm import tqdm

from .utils import LOG_INFO,correctPadding,stripPads
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
class BHOCR(object):    
    def __init__(self,
                model_path,
                img_height = 64,
                img_width  = 512,
                backbone   = 'densenet121',
                easy_ocr_gpu=False,
                use_tesseract=False):
        '''
            creates a BHOCR object
            args:
                model path  :   the path for "finetuned.h5"
                img_height  :   modifier model image height
                img_width   :   modifier model image width
                backbone    :   backbone for modifier
                easy_ocr_gpu:   use gpu for using easy ocr
                use_tesseract:  compare tesseract results with easy ocr
        '''
        self.img_height = img_height
        self.img_width  = img_width
        self.use_tesseract=use_tesseract

        LOG_INFO("Initializing modifier")
        self.modifier= sm.Unet(backbone,input_shape=( img_height , img_width,1), classes=1,encoder_weights=None)
        self.modifier.load_weights(model_path)
        LOG_INFO("Weights initialized")
        
        LOG_INFO("Initializing Recognizer:EasyOCR")
        self.easyOCR= easyocr.Reader(['bn'],gpu = easy_ocr_gpu)

        
    def infer(self,data,debug=False):
        '''
            infers on a word by word basis
            args:
                data    :   path of image to predict/ a numpy array
        '''
        if type(data)==str:
            # process word image
            data=cv2.imread(data,0)
        
        blur = cv2.GaussianBlur(data,(5,5),0)
        _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img=img-255
        img=stripPads(img,0)

        if debug:
            plt.imshow(img)
            plt.show()
        
        # resize (height based)
        h,w=img.shape 
        width= int(self.img_height* w/h) 
        img=cv2.resize(img,(width,self.img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        
        # pad correction
        img=correctPadding(img)
        
        # prediction
        data=np.expand_dims(img,axis=0)
        data=data/255.0
        pred= self.modifier.predict(data)[0]
        img=np.squeeze(pred)
        img=img*255
        img=img.astype("uint8")

        if debug:
            plt.imshow(img)
            plt.show()

        res=self.easyOCR.readtext(img,detail=0)
        print("EasyOCR Recognition:",res)
        if self.use_tesseract:
            res = pytesseract.image_to_string(img, lang='ben', config='--psm 6')
            print("Tesseract Recognition:",res)
        