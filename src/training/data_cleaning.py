import os,sys
path = os.path.abspath("src")
sys.path.append(path)

from utils.logger import Logger
logger = Logger(training=True).create_logger()

import numpy as np

class DataCleaner():
    def __init__(self,data):
        self.__data = data
        logger.info("Data cleaning initialized")
        
    def __removeOutliers(self,data):
        logger.info("Detecting Outliers")
        numerical_columns = data.select_dtypes('int').columns
        data = data[data["previous"]<=20]
        
        for column in numerical_columns:
            if column=="previous":
                continue
            
            q1 = np.percentile(data[column],25)
            q3 = np.percentile(data[column],75)
            iqr = q3-q1
            min_val = q1 - iqr*1.5
            max_val = q3 + iqr*1.5
                
            data = data[(min_val<=data[column]) & (data[column]<=max_val)]
        
        logger.info("Removed Outliers")
        return data
        
    def cleanData(self):
        self.__data.drop(columns=['pdays'],inplace=True)
        self.__data["poutcome"] = self.__data["poutcome"].replace('other','unknown')
        self.__data = self.__removeOutliers(self.__data)
        logger.info("Data cleaning completed")
        return self.__data
        