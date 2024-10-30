import os,sys
path = os.path.abspath("src")
sys.path.append(path)

from utils.logger import Logger
logger = Logger(training=True).create_logger()

from utils.constant import CONFIG_FILE_PATH
from utils.read_files import read_yaml

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pickle import dump

import pandas as pd

class ModelTrainer():
    
    best_model = None
    best_score = -1
    __best_model_name = None
    
    def __init__(self,data):
        self.__config = read_yaml(CONFIG_FILE_PATH)
        self.__data = data
        self.__models =  {
            "LogisticRegression":LogisticRegression(),
            "KNeighborsClassifier":KNeighborsClassifier(),
            "DecisionTreeClassifier":DecisionTreeClassifier(),
            "RandomForestClassifier":RandomForestClassifier(),
            "SVC":SVC(),
            "GaussianNB":GaussianNB(),
            "GradientBoostingClassifier":GradientBoostingClassifier(),
            "MLPClassifier":MLPClassifier(),
            "XGBClassifier":XGBClassifier(),
            "CatBoostClassifier":CatBoostClassifier(),
            "LGBMClassifier":LGBMClassifier()  
        }
        
        logger.info("Model Trainer Initialized")
    
    def __balanceData(self,data):
        x_train,y_train = data.iloc[:,:-1],data.iloc[:,-1]
        smote = SMOTE(sampling_strategy='auto')
        x,y = smote.fit_resample(x_train, y_train)
        
        logger.info("Balanced the Imbalanced data completion")
        return x,y
    
    def train(self):
        x,y = self.__balanceData(self.__data)
        
        f1_scores = {"model":[],"f1 score":[]}
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y)

        for name,instance in self.__models.items():
            logger.info(f"== {name} training started ==")
            
            instance.fit(x_train,y_train)
            predictions = instance.predict(x_test)
            f1 = f1_score(y_test,predictions)
            
            if f1>self.best_score:
                self.best_score = f1
                self.best_model = instance
                self.__best_model_name = name
                
            f1_scores["model"].append(name)
            f1_scores["f1 score"].append(f1)
            dump(instance,open(f"{self.__config.models.base_model}/{name}.pkl","wb"))
            
            logger.info(f"{name} training completed...")

        f1_scores_df = pd.DataFrame(data=f1_scores).sort_values(by = 'f1 score',ascending=False)
        f1_scores_df.to_csv(self.__config.models.results,index=False)

        dump(self.best_model,open(f"{self.__config.models.best_model_root}/{self.__best_model_name}.pkl","wb"))
        logger.info(f"Saved Best Model {self.__best_model_name}")
        
        return f1_scores_df