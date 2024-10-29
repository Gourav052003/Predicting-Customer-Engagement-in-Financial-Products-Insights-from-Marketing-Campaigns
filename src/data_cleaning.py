class DataCleaner:
    
    def __init__(self, data):
        self.__data = data
    
        
    def cleanData(self):
        self.__data.drop(columns=['pdays'],axis=1,inplace=True)
        self.__data['poutcome']=self.__data['poutcome'].replace('other','unknown')
        
        return self.__data
