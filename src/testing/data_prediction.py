import os,sys
path = os.path.abspath("src")
sys.path.append(path)

from pickle import load
from utils.constant import CONFIG_FILE_PATH
from utils.read_files import read_yaml
from utils.logger import Logger
logger = Logger(testing=True).create_logger()

class DataPrediction():
    
    def __init__(self,data):
        config = read_yaml(CONFIG_FILE_PATH)
        self.__data = data 
        self.__best_model = load(open(config.models.best_model,"rb"))
        logger.info("Data Predictor initialized")
    
    def predict(self):
        predictions = self.__best_model.predict(self.__data)
        logger.info("Data Predictions Completed")
        return predictions
    