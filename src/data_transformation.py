import numpy as np
import pandas as pd
from pickle import load

class DataTransformer():
    
    def __init__(self,data):
        self.__data = data
        
        self.__scalers = {
            "age":load(open(f"artifacts/min-max-scalers/age.pkl","rb")),
            "balance":load(open(f"artifacts/min-max-scalers/balance.pkl","rb")),
            "duration":load(open(f"artifacts/min-max-scalers/duration.pkl","rb"))
        }

        self.__one_hot_encoders = {
            "marital":load(open(f"artifacts/one-hot-encoders/marital.pkl","rb")),
            "contact":load(open(f"artifacts/one-hot-encoders/contact.pkl","rb"))   
        }

        self.__cat_boost_encoders = {
            "job":load(open(f"artifacts/cat-boost-encoders/job.pkl","rb"))
        }
        
    
    def transform(self):
        self.__data["default"] = self.__data["default"].map({'no':0,"yes":1})
        self.__data["housing"] = self.__data["housing"].map({'no':0,"yes":1})
        self.__data["loan"] = self.__data["loan"].map({'no':0,'yes':1})
       
        self.__data["education"] = self.__data["education"].map({'unknown':-1,'primary':1, 'secondary':2,'tertiary':3})
        self.__data["poutcome"] = self.__data["poutcome"].map({"unknown":-1,"failure":0,"success":1})

        month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        self.__data['month'] = self.__data['month'].map(month_order)
        self.__data['month'] = np.sin(2 * np.pi * (self.__data['month']-1) / 12)
        self.__data['day'] = np.cos(2 * np.pi * (self.__data['day']-1) / 31)

        self.__data["campaign"] = np.log1p(self.__data["campaign"])
        self.__data["previous"] = np.log1p(self.__data["previous"])
        
        
        for feature,scaler in self.__scalers.items():
            self.__data[feature] = scaler.transform(self.__data[[feature]])
        
        for feature,one_hot_encoder in self.__one_hot_encoders.items():
            encoded_data = one_hot_encoder.transform(self.__data[[feature]])
            encoded_df = pd.DataFrame(encoded_data,columns=[f"{feature}_0",f"{feature}_1"])
            self.__data = pd.concat([encoded_df,self.__data],axis=1)
            self.__data.drop([feature],inplace=True,axis=1)
        
        catboost_encoder = self.__cat_boost_encoders["job"]
        self.__data["job"] = catboost_encoder.transform((self.__data["job"]))
            
        return self.__data