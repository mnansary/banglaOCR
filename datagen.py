# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import os 
import json
import random
import cv2 
import pandas as pd
import numpy as np 
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 
from tqdm import tqdm
from glob import glob
from ast import literal_eval
from coreLib.dataset import DataSet
from coreLib.utils import create_dir,correctPadding,stripPads,LOG_INFO,lambda_paded_label
from coreLib.words import single
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):


    filename=[]
    labels  =[]
    _path   =[]
    data_path   =   args.data_path
    main_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    nb_train    =   int(args.nb_train)
    max_word_length =   int(args.max_word_length)+1
    
    # dataset object
    ds=DataSet(data_path)
    main_path=create_dir(main_path,"segCRNNdata")
    
    
    # pairs
    save_path=create_dir(main_path,"bs")
    img_dir=create_dir(save_path,"images")
    tgt_dir=create_dir(save_path,"targets")
    map_dir=create_dir(save_path,"maps")
    seg_dir=create_dir(save_path,"segocr_outs")


    # data
    df=ds.boise_state.df
    df["img_path"]=df.filename.progress_apply(lambda x: os.path.join(ds.boise_state.dir,x))

    
    font_path     =os.path.join(ds.bangla.fonts,"Bangla.ttf")
    # create font
    font=PIL.ImageFont.truetype(font_path, size=img_height)

    

    bs_skip=[]
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
            # target and map
            #-----------------
            # unique values
            vals=list(np.unique(img))
            # construct target    
            tgts=[]
            maps=[]
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

                    #--------------------------------
                    # maps
                    #--------------------------------
                    h,w=tgt.shape
                    map=np.zeros((h,w))
                    map[int(h/4):int(3*h/4),int(w/4):int(3*w/4)]=ds.vocab.index(comp)
                    maps.append(map)


            tgt=np.concatenate(tgts,axis=1)
            map=np.concatenate(maps,axis=1)
            # resize
            h,w=img.shape 
            tgt=cv2.resize(tgt,(w,h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)

            # revalue
            img[img<255]=0
            tgt=255-tgt
            # pad correction
            img=correctPadding(img,dim=(img_height,img_width))
            tgt=correctPadding(tgt,dim=(img_height,img_width))
            map=correctPadding(map,dim=(img_height,img_width),pad_val=0)
            h,w=map.shape
            seg=cv2.resize(map,(w,h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            
            # save
            cv2.imwrite(os.path.join(img_dir,f"bs{idx}.png"),img)
            cv2.imwrite(os.path.join(tgt_dir,f"bs{idx}.png"),tgt)
            np.save(os.path.join(map_dir,f"bs{idx}.npy"),map)
            np.save(os.path.join(seg_dir,f"bs{idx}.npy"),seg)

            filename.append(f"bs{idx}")
            labels.append(comps)
            _path.append(os.path.join(img_dir,f"bs{idx}.png"))
            
        except Exception as e:
            #LOG_INFO(e)
            bs_skip.append(idx)
    
    LOG_INFO(f"skipped:{len(bs_skip)}")


    # pairs
    save_path=create_dir(main_path,"synth")
    img_dir=create_dir(save_path,"images")
    tgt_dir=create_dir(save_path,"targets")
    map_dir=create_dir(save_path,"maps")
    seg_dir=create_dir(save_path,"segocr_outs")

    # create the images
    for i in tqdm(range(nb_train)):
        try:
            # selection
            comp_type =random.choice(["grapheme"])
            use_dict  =random.choice([True,False])
            img,tgt,map,label=single(ds,comp_type,use_dict,(img_height,img_width))
            h,w=map.shape
            seg=cv2.resize(map,(w,h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            
            
            # save
            cv2.imwrite(os.path.join(img_dir,f"synth{i}.png"),img)
            cv2.imwrite(os.path.join(tgt_dir,f"synth{i}.png"),tgt)
            np.save(os.path.join(map_dir,f"synth{i}.npy"),map)
            np.save(os.path.join(seg_dir,f"synth{idx}.npy"),seg)

            filename.append(f"synth{i}")
            labels.append(label)
            _path.append(os.path.join(img_dir,f"synth{i}.png"))
            
        except Exception as e:
            print(e)

    # create dataframe
    df_s=pd.DataFrame({"filename":filename,"labels":labels,"img_path":_path})
    # length
    df_s["label_len"]=df_s.labels.progress_apply(lambda x:len(x))
    # label_lenght correction
    df_s=df_s.loc[df_s.label_len<max_word_length]
    # encode
    df_s["encoded"]= df_s.labels.progress_apply(lambda x:[ds.vocab.index(i) for i in x])
    df_s["glabel"] = df_s.encoded.progress_apply(lambda x:lambda_paded_label(x,max_word_length))
    
    # save
    df_s.to_csv(os.path.join(main_path,"data.csv") ,index=False)
    





    # config 
    config={'img_height':img_height,
            'img_width':img_width,   
            'nb_channels':3,
            'vocab':ds.vocab,
            'synthetic_data':nb_train,
            'boise_state_data':len(df),
            'max_word_len':max_word_length,
            'map_size':int(img_height*img_width),
            'seg_size':int(img_height//2*img_width//2)
            }
    config_json=os.path.join(main_path,"config.json")
    with open(config_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)
    
  
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("BHOCR Pre-Training Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the input data folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=32,help ="height for each grapheme: default=32")
    parser.add_argument("--img_width",required=False,default=128,help ="width dimension of word images: default=128")
    parser.add_argument("--nb_train",required=False,default=100000,help ="number of images for training:default:100000")
    parser.add_argument("--max_word_length",required=False,default=10,help ="maximum word lenght data to keep:default:10")
    
    args = parser.parse_args()
    main(args)
    
    
