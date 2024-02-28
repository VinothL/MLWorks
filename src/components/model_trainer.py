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
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score

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
                "Random Forest Classifier":{'max_depth':[3,5,10,None],
                                            'n_estimators':[5,10,15],
                                            'max_features':[1,3,5,7],
                                            'min_samples_leaf':[100,200,300],
                                            'min_samples_split':[1000,2000,3000]
                                          },
                "Ada Boost Classifier":{},
                "XGBoost Classifier":{
                                           'min_child_weight': [1, 5, 10],
                                           'gamma': [0.5, 1, 1.5, 2, 5],
                                           'subsample': [0.6, 0.8, 1.0],
                                           'colsample_bytree': [0.6, 0.8, 1.0],
                                           'max_depth': [3, 4, 5]
                                     }
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
                resultSummary.setdefault(modelName,{})['model_name'] = modelName
                resultSummary.setdefault(modelName,{})['model_obj'] = model
                resultSummary.setdefault(modelName,{})['gs_params'] = gs.best_params_
                resultSummary.setdefault(modelName,{})['accuracy'] = accuracy
                resultSummary.setdefault(modelName,{})['precision'] = precision
                resultSummary.setdefault(modelName,{})['recall'] = recall
                resultSummary.setdefault(modelName,{})['f1score'] = f1score
                resultSummary.setdefault(modelName,{})['auc'] = auc

            logging.info(resultSummary)
            return resultSummary

        except Exception as e:
            raise customException(e, sys)
    
    def getBestModel(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test,models,params = self.initModelTrainer(train_array,test_array)
            resultSummary = self.runModelTraining(X_train,y_train,X_test,y_test,models,params)
            bestAUCScore = sorted(resultSummary.items(), key=lambda item: item[1]['auc'],reverse=True)[1][1]['auc']
            gsParams = sorted(resultSummary.items(), key=lambda item: item[1]['auc'],reverse=True)[1][1]['gs_params']
            bestModel = sorted(resultSummary.items(), key=lambda item: item[1]['auc'],reverse=True)[1][1]['model_obj']
            bestModelName =  sorted(resultSummary.items(), key=lambda item: item[1]['auc'],reverse=True)[1][1]['model_name']
            logging.info("Best Model :" + str(bestModelName))
            logging.info("selected the best auc :" + str(bestAUCScore))

            if bestAUCScore < 0.5:
                raise customException("No best model found")
                logging.info("No best model found")
            
            logging.info("Best model found")
            
            saveObject(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=bestModel
            )

            predicted=bestModel.predict(X_test)
            aucScore = roc_auc_score(y_test, predicted)
            logging.info('Test AUC Score for the best model :' + str(aucScore))

            predicted=models[bestModelName].predict(X_test)
            aucScoreBase = roc_auc_score(y_test, predicted)
            logging.info('Test AUC Score for the best model wo gs:' + str(aucScoreBase))


            return aucScore

        except Exception as e:
            raise customException("No Model meeting the criteria")

if __name__ =='__main__':
    di = DataIngestion()
    train_path,test_path = di.runDataIngestion()

    dt = DataTransformation()
    train_arr,test_arr,prePipePath = dt.runPreprocessor(train_path,test_path)
    
    mt = ModelTrainer()
    aucScore = mt.getBestModel(train_arr,test_arr)
    
    

