import os,sys
path = os.path.abspath("src")
sys.path.append(path)

import numpy as np
import pandas as pd
from pickle import dump
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from category_encoders.cat_boost import CatBoostEncoder

from utils.constant import CONFIG_FILE_PATH
from utils.read_files import read_yaml
from utils.logger import Logger
logger = Logger(training=True).create_logger()

class DataTransformer():
    
    def __init__(self,data):
        self.__config = read_yaml(CONFIG_FILE_PATH)
        self.__data = data
        
        self.__min_max_scalers = {
            "age":MinMaxScaler(),
            "balance":MinMaxScaler(),
            "duration":MinMaxScaler()
        }

        self.__one_hot_encoders = {
            "marital":OneHotEncoder(drop='first',sparse_output=False),
            "contact":OneHotEncoder(drop='first',sparse_output=False)    
        }

        self.__cat_boost_encoders = {
            "job":CatBoostEncoder()
        }
        
        logger.info("Data Transformer Initialized")
        
    
    def transform(self):
        self.__data["default"] = self.__data["default"].map({'no':0,"yes":1})
        self.__data["housing"] = self.__data["housing"].map({'no':0,"yes":1})
        self.__data["loan"] = self.__data["loan"].map({'no':0,'yes':1})
        self.__data["y"] = self.__data["y"].map({'no':0,'yes':1})
       
        self.__data["education"] = self.__data["education"].map({'unknown':-1,'primary':1, 'secondary':2,'tertiary':3})
        self.__data["poutcome"] = self.__data["poutcome"].map({"unknown":-1,"failure":0,"success":1})

        month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        self.__data['month'] = self.__data['month'].map(month_order)
        self.__data['month'] = np.sin(2 * np.pi * (self.__data['month']-1) / 12)
        self.__data['day'] = np.cos(2 * np.pi * (self.__data['day']-1) / 31)

        self.__data["campaign"] = np.log1p(self.__data["campaign"])
        self.__data["previous"] = np.log1p(self.__data["previous"])
    
        logger.info("min_max_scaler encoding")
        for feature,min_max_scaler in self.__min_max_scalers.items():
            min_max_scaler.fit(self.__data[[feature]])
            self.__data[feature] = min_max_scaler.transform(self.__data[[feature]])
            dump(min_max_scaler,open(f"{self.__config.min_max_scalers.root}/{feature}.pkl",'wb'))
                
        logger.info("one hot encoding")
        for feature,one_hot_encoder in self.__one_hot_encoders.items():
            logger.info(f"1., {self.__data['contact'].unique()},{self.__data.shape}")
            one_hot_encoder.fit(self.__data[[feature]])
            encoded_data = one_hot_encoder.transform(self.__data[[feature]])
            encoded_df = pd.DataFrame(encoded_data,columns=[f"{feature}_0",f"{feature}_1"])
            logger.info(encoded_df.shape)
            logger.info(f"2., {self.__data['contact'].unique()},,{self.__data.shape}")
            self.__data.reset_index(inplace=True,drop=True)
            self.__data = pd.concat([encoded_df,self.__data],axis=1)
            # self.__data = encoded_df.join(self.__data)
            logger.info(f"3. {self.__data['contact'].unique()},,{self.__data.shape}")
            self.__data.drop([feature],inplace=True,axis=1)
            # self.__data.dropna(inplace=True)
            
            dump(one_hot_encoder,open(f"{self.__config.one_hot_encoders.root}/{feature}.pkl",'wb'))
          
        logger.info("catboost encoding")
        catboost_encoder = self.__cat_boost_encoders["job"]
        catboost_encoder.fit(self.__data["job"],self.__data["y"])
        self.__data["job"] = catboost_encoder.transform((self.__data["job"]))
        dump(catboost_encoder,open(f"{self.__config.cat_boost_encoders.root}/job.pkl",'wb'))
        logger.info("Data Transformation Completed")    
          
        return self.__data