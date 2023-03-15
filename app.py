
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,Modeltrainer
from src.components.data_ingestion import DataIngestion

obj=DataIngestion()
train_data,test_data=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

model_trainer=Modeltrainer()
print(model_trainer.initiate_model_trainer(train_arr,test_arr))