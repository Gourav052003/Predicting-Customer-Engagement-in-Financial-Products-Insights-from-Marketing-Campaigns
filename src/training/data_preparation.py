import os,sys
path = os.path.abspath("src")
sys.path.append(path)

import pandas as pd
from utils.logger import Logger
logger = Logger(training=True).create_logger()

class DataPreparer():
    
    def __init__(self,data):
        self.__data = data
        logger.info("Data Preparation initialized")
    
    def getData(self):
        self.__data = pd.read_csv(self.__data)
        self.__data.to_csv('train.csv',index=False)
        logger.info(self.__data["contact"].unique())
        logger.info("Data Preparation completed")
        return self.__data