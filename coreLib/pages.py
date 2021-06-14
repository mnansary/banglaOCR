# # -*-coding: utf-8 -
# '''
#     @author: MD. Nazmuddoha Ansary
# '''
# #--------------------
# # imports
# #--------------------
# import os 
# import json
# import cv2
# import numpy as np
# import pandas as pd 
# import string
# import random

# from glob import glob
# from tqdm.auto import tqdm
# from .utils import *
# tqdm.pandas()
# #--------------------
# # GLOBALS
# #--------------------
# # symbols to avoid 
# SYMBOLS = ['`','~','!','@','#','$','%',
#            '^','&','*','(',')','_','-',
#            '+','=','{','[','}','}','|',
#            '\\',':',';','"',"'",'<',
#            ',','>','.','?','/',
#            '১','২','৩','৪','৫','৬','৭','৮','৯','০',
#            '।']
# SYMBOLS+=list(string.ascii_letters)
# SYMBOLS+=[str(i) for i in range(10)]
# GP=GraphemeParser()
# #--------------------------------images2words------------------------------------------------------------
# #--------------------
# # helper functions
# #--------------------

# def extract_word_images_and_labels(img_path):
#     '''
#         extracts word images and labels from a given image
#         args:
#             img_path : path of the image
#         returns:
#             (images,labels)
#             list of images and labels
#     '''
#     imgs=[]
#     labels=[]
#     # json_path
#     json_path=img_path.replace("jpg","json")
#     # read image
#     data=cv2.imread(img_path,0)
#     # label
#     label_json = json.load(open(json_path,'r'))
#     # get word idx
#     for idx in range(len(label_json['shapes'])):
#         # label
#         label=str(label_json['shapes'][idx]['label'])
#         # special charecter negation
#         if not any(substring in label for substring in SYMBOLS):
#             labels.append(label)
#             # crop bbox
#             xy=label_json['shapes'][idx]['points']
#             # crop points
#             x1 = int(np.round(xy[0][0]))
#             y1 = int(np.round(xy[0][1]))
#             x2 = int(np.round(xy[1][0]))
#             y2 = int(np.round(xy[1][1]))
#             # image
#             img=data[y1:y2,x1:x2]
#             imgs.append(img)
#     return imgs,labels

    
# #--------------------
# # ops
# #--------------------
# def pages2words(ds,
#                 dim=(128,32),
#                 split_perc=20,
#                 label_sep=False):
#     '''
#         creates the images based on labels
#         args:
#             ds            :  dataset object
#             split_perc    :  test split perc
#             dim           :  (img_width,img_height) tuple to resize to 
#             label_sep     :  name separated with label
#     '''
#     img_idens=[]
#     img_labels=[]
#     src_imgs=[]
#     i=0
#     save_path=ds.word_path
#     LOG_INFO(save_path)
#     # get image paths
#     img_paths=[img_path for img_path in glob(os.path.join(ds.pages,"*.jpg"))]
#     # iterate
#     for img_path in tqdm(img_paths):
#         # extract images and labels
#         imgs,labels=extract_word_images_and_labels(img_path)
#         if len(imgs)>0:
#             for img,label in zip(imgs,labels):
#                 try:
#                     # resize to char dim
#                     img=cv2.resize(img,dim)
#                     # save path for the word
#                     if label_sep:
#                         img_save_path=os.path.join(save_path,f"{label}_{i}.png")
#                     else:
#                         img_save_path=os.path.join(save_path,f"{i}.png")
#                     # save
#                     cv2.imwrite(img_save_path,img)
#                     # append
#                     img_idens.append(f"{i}.png")
#                     img_labels.append(label)
#                     src_imgs.append(os.path.basename(img_path))

#                     i=i+1
                    
#                 except Exception as e: 
#                     LOG_INFO(f"error in creating image:{img_path} label:{label},error:{e}",mcolor='red')
#     # save to csv
#     df=pd.DataFrame({"image_id":img_idens,"label":img_labels,"src":src_imgs})
#     # graphemes
#     df["graphemes"]=df.label.progress_apply(lambda x:GP.word2grapheme(x))
#     # cleanup
#     df.graphemes=df.graphemes.progress_apply(lambda x: x if set(x)<=set(ds.known_graphemes) else None)
#     df.dropna(inplace=True)
#     # unicodes
#     df["unicodes"]=df.label.progress_apply(lambda x:[i for i in x])
    
#     # char vocab
#     symbol_lists=df.unicodes.tolist()
#     cvocab  = get_sorted_vocab(symbol_lists)
#     max_len= max([len(l) for l in symbol_lists])
#     df["clabel"]= df.unicodes.progress_apply(lambda x: get_encoded_label(x,cvocab))
#     df.clabel   = df.clabel.progress_apply(lambda x: pad_encoded_label(x,max_len,len(cvocab)))

#     # grapheme vocab
#     symbol_lists=df.graphemes.tolist()
#     gvocab  = get_sorted_vocab(symbol_lists)
#     max_len= max([len(l) for l in symbol_lists])
#     df["glabel"]= df.graphemes.progress_apply(lambda x: get_encoded_label(x,gvocab))
#     df.glabel   = df.glabel.progress_apply(lambda x: pad_encoded_label(x,max_len,len(gvocab)))    
    
#     # test train split
#     srcs=list(df.src.unique())
#     random.shuffle(srcs)
#     test_len=int(len(srcs)*split_perc/100)
#     test_srcs=srcs[:test_len]
#     df["mode"]=df.src.progress_apply(lambda x: "test" if x in test_srcs else "train")
    
#     # img_path
#     df["img_path"]=df.image_id.progress_apply(lambda x: os.path.join(save_path,x))

    
#     class word:
#         # pure word data
#         data=df.drop_duplicates(subset=['label'])
#         data=data[["graphemes","clabel","glabel"]]
        
#         # split of test and train
#         train=df.loc[df["mode"]=="train"]
#         train=train[["img_path","clabel","glabel"]]
#         test=df.loc[df["mode"]=="test"]
#         test=test[["img_path","clabel","glabel"]]
    
#     class vocab:
#         charecter=cvocab
#         grapheme =gvocab

#     ds.word=word 
#     ds.vocab=vocab 
    
#     return ds