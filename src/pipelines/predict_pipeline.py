import os 
import sys
import pandas as pd

from src.exception import customException
from src.utils import loadObject
from src.logger import logging 


class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self,features):
        try:
            model_path= os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocess_pipeline.pkl')

            model=loadObject(file_path=model_path)
            preprocessor=loadObject(file_path=preprocessor_path)
            logging.info("Preprocessor and Model Loaded")
            
            logging.info("logging the features")
            logging.info(features)
                        
            data_scaled=preprocessor.transform(features)
            logging.info("Data Transformation Completed")

            preds=model.predict(data_scaled)
            logging.info("Prediction completed for the Customer")

            return preds
        except Exception as e:
            raise customException(e,sys)
        

    

class CustomData:
    '''
    Responsible for mapping all the data points we receive in the front end to the back end.
    '''
    def __init__(self,
                 CreditScore: float,
                 Tenure : int,
                 NumOfProducts: int,
                 Geography: str,
                 Gender:str) -> None:
        
        self.CreditScore = CreditScore
        self.Tenure = Tenure
        self.NumOfProducts = NumOfProducts
        self.Geography = Geography
        self.Gender = Gender
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.CreditScore],
                "Tenure": [self.Tenure],
                "NumOfProducts": [self.NumOfProducts],
                "Geography": [self.Geography],
                "Gender": [self.Gender]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise customException(e, sys)