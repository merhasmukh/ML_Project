import os 
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import  dill
from src.logger import logging
from sklearn.metrics import r2_score
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train, y_train,X_test,y_test,models):
    try:
        report={}
        logging.info("Evaluating Models Started {}".format(list(models)))
        logging.info("X_train shape {} y_train shape {}".format(X_train.shape,y_train.shape))
        for i in range(len(list(models))):
            # print("iii",i)
            model = list(models.values())[i]
            # print("model",model)
            model.fit(X_train, y_train) # Train model
            logging.info("{} fitted sucessfully".format(model))
            
            # Make predictions
            # y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            logging.info("Test model score for {} is {}".format(model,test_model_score))
            
            # print(test_model_score)

            report[list(models.keys())[i]]=test_model_score
        # print(report)
        return report

    except:pass