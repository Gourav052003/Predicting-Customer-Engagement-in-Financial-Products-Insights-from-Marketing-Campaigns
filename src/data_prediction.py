from pickle import load

class DataPrediction():
    
    def __init__(self,data):
        self.__data = data 
        self.__best_model = load(open(f"models/base models/CatBoostClassifier.pkl","rb"))
    
    def predict(self):
        predictions = self.__best_model.predict(self.__data)
        return predictions
    