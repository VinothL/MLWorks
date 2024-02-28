import os 
import sys 

from src.exception import customException
from src.logger import logging 
from src.utils import saveObject

from data_ingestion import DataIngestion

import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from dataclasses import dataclass 

@dataclass
class DataTransformationConfig():
    preprocessor_pipeline_file_path = os.path.join('artifacts','preprocess_pipeline.pkl')

class DataTransformation():
    def __init__(self):
        self.preprocessorPipeConfig = DataTransformationConfig()

    def getPreprocessorPipeline(self):
        """
        This function has the logic to built the required preprocessor pipeline for both
        numerical and categorical.
        """
        try: 
            numCols = ['CreditScore', 'Tenure', 'NumOfProducts']
            catCols = ['Geography', 'Gender']

            logging.info('Numerical Columns :' + str(numCols))
            logging.info('Categorical Columns :' + str(catCols))

            numPipe = Pipeline(
                steps =[
                    ("scaler",StandardScaler())
                ]
            )

            catPipe = Pipeline(
                steps =[
                    ("OneHot",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessorPipe = ColumnTransformer(
                [
                    ("numPipeline",numPipe,numCols),
                    ("catPipeline",catPipe,catCols)
                ]
            )

            return preprocessorPipe
    
        except Exception as e: 
            raise customException(e,sys)


    def runPreprocessor(self,train_path,test_path):
        try:
            logging.info("Loading the train and test datasets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            targetColName = 'Exited'
            trainX = train_df.drop(columns = [targetColName], axis=1)
            trainY = train_df[targetColName]

            testX = train_df.drop(columns = [targetColName], axis=1)
            testY = train_df[targetColName]

            logging.info("Preprocessing started")
            preprocessorPipe = self.getPreprocessorPipeline()

            trainX_arr = preprocessorPipe.fit_transform(trainX)
            testX_arr = preprocessorPipe.transform(testX)

            train_arr = np.c_[trainX_arr,np.array(trainY)]
            test_arr = np.c_[testX_arr,np.array(testY)]

            logging.info("Preprocessing completed")

            saveObject (
                file_path = self.preprocessorPipeConfig.preprocessor_pipeline_file_path,
                obj = preprocessorPipe
            )

            return (
                train_arr,
                test_arr,
                self.preprocessorPipeConfig.preprocessor_pipeline_file_path
            )


        except Exception as e:
            raise customException(e,sys)

if __name__ == "__main__":
    di = DataIngestion()
    train_path,test_path = di.runDataIngestion()

    dt = DataTransformation()
    train_arr,test_arr,prePipePath = dt.runPreprocessor(train_path,test_path)









