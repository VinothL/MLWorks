import os 
import sys

from src.exception import customException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str =  os.path.join('artifacts','raw_data.csv')
    train_data_path: str =  os.path.join('artifacts','train_data.csv')
    test_data_path: str = os.path.join('artifacts','test_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Data Ingestion Config created")
    
    def runDataIngestion(self):
        logging.info('Data Ingestion Starts')
        try:
            df = pd.read_csv('Notebooks/data/processed_data.csv')

            logging.info('Creating the artifacts folder')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Initiating the train and test split')
            train_set,test_set = train_test_split(df,test_size = 0.2,random_state = 42)

            logging.info('Writing the train and test datasets')
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header = True )
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header = True )

            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            

        except Exception as e:
            raise customException(e,sys)



if __name__ == "__main__":
    logging.info('Executing the DataIngestion Test from Main')
    di =  DataIngestion()
    train_path,test_path = di.runDataIngestion()







