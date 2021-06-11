# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd 
from glob import glob
from tqdm import tqdm
from ast import literal_eval
from .utils import LOG_INFO
tqdm.pandas()
#--------------------
# class info
#--------------------
class DataSet(object):
    def __init__(self,data_dir):
        '''
            data_dir : the location of the data folder
        '''
        self.data_dir=data_dir
        
        class bangla:
            class graphemes:
                dir   =   os.path.join(data_dir,"bangla","graphemes")
                csv   =   os.path.join(data_dir,"bangla","graphemes.csv")
            class numbers:
                dir   =   os.path.join(data_dir,"bangla","numbers")
                csv   =   os.path.join(data_dir,"bangla","numbers.csv")
            dict_csv  =   os.path.join(data_dir,"bangla","dictionary.csv")
            fonts     =   os.path.join(data_dir,"bangla","fonts")
        
        

        # assign
        self.bangla  = bangla
        # error check
        self.__checkExistance()

        # get dfs
        self.bangla.graphemes.df=self.__getDataFrame(self.bangla.graphemes)
        self.bangla.numbers.df  =self.__getDataFrame(self.bangla.numbers)
        self.bangla.dictionary  =self.__getDataFrame(self.bangla.dict_csv,is_dict=True)

        
        # data validity
        self.__checkDataValidity(self.bangla.graphemes,"bangla.graphemes")
        self.__checkDataValidity(self.bangla.numbers,"bangla.numbers")
        self.__checkDataValidity(self.bangla.fonts,"bangla.fonts",check_dir_only=True)
        
        
        
        
        
        
        
    def __getDataFrame(self,obj,is_dict=False,int_label=False):
        '''
            creates the dataframe from a given csv file
            args:
                obj       =   csv file path or class 
                is_dict   =   only true if the given is a dictionary 
                int_label =   if the labels are int convert string
        '''
        try:
            
            if is_dict:
                df=pd.read_csv(obj)
                assert "word" in df.columns,f"word column not found:{obj}"
                assert "graphemes" in df.columns,f"graphemes column not found:{obj}"
                 
                LOG_INFO(f"Processing Dictionary:{obj}")
                df.graphemes=df.graphemes.progress_apply(lambda x: literal_eval(x))
            else:
                csv=obj.csv
                img_dir=obj.dir
                df=pd.read_csv(csv)
                assert "filename" in df.columns,f"filename column not found:{csv}"
                assert "label" in df.columns,f"label column not found:{csv}"
                df["img_path"]=df["filename"].progress_apply(lambda x:os.path.join(img_dir,f"{x}.bmp"))
                if int_label:
                    LOG_INFO("converting int labels to string")
                    df.label=df.label.progress_apply(lambda x: str(x))
            return df
        except Exception as e:
            LOG_INFO(f"Error in processing:{csv}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red") 
                

    def __checkDataValidity(self,obj,iden,check_dir_only=False):
        '''
            checks that a folder does contain proper images
        '''
        try:
            LOG_INFO(iden)
            if check_dir_only:
                data=[data_path for data_path in tqdm(glob(os.path.join(obj,"*.*")))]
                assert len(data)>0, f"No data paths found({iden})"
            else:
                imgs=[img_path for img_path in tqdm(glob(os.path.join(obj.dir,"*.*")))]
                assert len(imgs)>0, f"No data paths found({iden})"
                assert len(imgs)==len(obj.df), f"Image paths doesnot match labels data({iden}:{len(imgs)}!={len(obj.df)})"
                
        except Exception as e:
            LOG_INFO(f"Error in Validity Check:{iden}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red")        


    def __checkExistance(self):
        '''
            check for paths and make sure the data is there 
        '''
        assert os.path.exists(self.bangla.graphemes.dir),"Bangla graphemes dir not found"
        assert os.path.exists(self.bangla.graphemes.csv),"Bangla graphemes csv not found"
        assert os.path.exists(self.bangla.numbers.dir),"Bangla numbers dir not found"
        assert os.path.exists(self.bangla.numbers.csv),"Bangla numbers csv not found"
        assert os.path.exists(self.bangla.fonts),"Bangla fonts not found"
        assert os.path.exists(self.bangla.dict_csv),"Bangla dictionary not found"
        
        
        LOG_INFO("All paths found",mcolor="green")
        
        
        
        
        


    
        