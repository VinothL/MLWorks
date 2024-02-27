import os 
import sys 

from src.exception import customException
from src.logger import logging
from utils import saveObject,evaluateModel
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier

from data_ingestion import DataIngestion
from data_transformation import DataTransformation


from dataclasses import dataclass
from collections import defaultdict



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initModelTrainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "Linear SVC": LinearSVC(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Ada Boost Classifier": AdaBoostClassifier(),
                "XGBoost Classifier": XGBClassifier()
                }
            
            params = {
                "Logistic Regression":{},
                "Linear SVC":{},
                "Decision Tree Classifier":{},
                "Random Forest Classifier":{},
                "Ada Boost Classifier":{},
                "XGBoost Classifier":{}
                    }
            
            return X_train,y_train,X_test,y_test,models,params
        
        except Exception as e:
            raise customException(e,sys)

    def runModelTraining(self,X_train,y_train,X_test,y_test,models,params):
        try:
            resultSummary = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para=params[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                accuracy,precision,recall,f1score,auc = evaluateModel(y_train, y_train_pred)
                accuracy,precision,recall,f1score,auc = evaluateModel(y_test, y_test_pred)

                modelName = str(list(models.keys())[i])
                resultSummary.setdefault(modelName,{})['accuracy'] = accuracy
                resultSummary.setdefault(modelName,{})['precision'] = precision
                resultSummary.setdefault(modelName,{})['recall'] = recall
                resultSummary.setdefault(modelName,{})['f1score'] = f1score
                resultSummary.setdefault(modelName,{})['auc'] = auc

            logging.info(resultSummary)
            return resultSummary

        except Exception as e:
            raise customException(e, sys)
    
    def getBestModel():
        try:
            

            pass
        except Exception as e:
            raise customException("No Model meeting the criteria",sys)

if __name__ =='__main__':
    di = DataIngestion()
    train_path,test_path = di.runDataIngestion()

    dt = DataTransformation()
    train_arr,test_arr,prePipePath = dt.runPreprocessor(train_path,test_path)
    
    mt = ModelTrainer()
    X_train,y_train,X_test,y_test,models,params = mt.initModelTrainer(train_arr,test_arr)
    resultSummary = mt.runModelTraining(X_train,y_train,X_test,y_test,models,params)

