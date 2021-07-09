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
    
    # resources
    rec_path=create_dir( os.path.join(rec_path,"segCRNNdata"),"tfrecords")
    config_json=os.path.join(data_path,"segCRNNdata","config.json")
    data_csv   =os.path.join(data_path,"segCRNNdata","data.csv")
    #------
    # config
    #------
    with open(config_json) as f:
        config = json.load(f)

    vocab=config["vocab"]



    df=pd.read_csv(data_csv)
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
    
    args = parser.parse_args()
    main(args)
    
    
