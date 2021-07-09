# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import pandas as pd
import os
import json
from ast import literal_eval
from tqdm.auto import tqdm
tqdm.pandas()

from coreLib.utils import create_dir,LOG_INFO
from coreLib.store import genTFRecords
#--------------------
# main
#--------------------
def main(args):
    data_path       =   args.data_path
    rec_path        =   args.save_path
    max_word_length =   int(args.max_word_length)+1
    
    # resources
    rec_path=create_dir(rec_path,"segCRNNdata","tfrecords")
    config_json=os.path.join(data_path,"segCRNNdata","config.json")
    data_csv   =os.path.join(data_path,"segCRNNdata","data.csv")
    #------
    # config
    #------
    with open(config_json) as f:
        config = json.load(f)

    vocab=config["vocab"]

    #------
    # data csv
    #------
    def create_paded_label(x,max_len):
        for _ in range(len(x),max_len):
            x.append(0)
        return x

    df=pd.read_csv(data_csv)
    df.labels=df.labels.progress_apply(lambda x:literal_eval(x))
    df["label_len"]=df.labels.progress_apply(lambda x:len(x))
    df=df.loc[df.label_len<max_word_length]
    df["encoded"]=df.labels.progress_apply(lambda x:[vocab.index(i) for i in x])
    df["glabel"]=df.encoded.progress_apply(lambda x:create_paded_label(x,max_word_length))
    df=df[["img_path","glabel"]]

    genTFRecords(df,rec_path)
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("BHOCR Dataset storing script")
    parser.add_argument("data_path", help="Path of the input data folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--max_word_length",required=False,default=12,help ="maximum word lenght data to keep:default:12")
    
    args = parser.parse_args()
    main(args)
    
    
