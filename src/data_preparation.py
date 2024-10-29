import pandas as pd

class DataPreparer():
       
    def __init__(self, age=30, balance=0, day=1, duration=60, campaign=1, 
                 pdays=-1, previous=0, job='unknown', marital='single', 
                 education='unknown', default='no', housing='no', loan='no', 
                 contact='unknown', month='jan', poutcome='unknown'):
        
        self.__data = pd.DataFrame({
            "age":[age],
            "job":[job],
            "marital":[marital],
            "education":[education],
            "default":[default],
            "balance":[balance],
            "housing":[housing],
            "loan":[loan],
            "contact":[contact],
            "day":[day],
            "month":[month],
            "duration":[duration],
            "campaign":[campaign],
            "pdays":[pdays],
            "previous":[previous],
            "poutcome":[poutcome]
        })
        
    def getData(self):
        return self.__data
    