import os,sys
path = os.path.abspath("src")
sys.path.append(path)

from utils.logger import Logger
logger = Logger(testing=True).create_logger()

class DataCleaner():
    
    def __init__(self, data):
        self.__data = data
        logger.info("Data Cleaner Initialized")
    
        
    def cleanData(self):
        self.__data.drop(columns=['pdays'],axis=1,inplace=True)
        self.__data['poutcome']=self.__data['poutcome'].replace('other','unknown')
        
        logger.info("Data Cleaning Completed")
        return self.__data
