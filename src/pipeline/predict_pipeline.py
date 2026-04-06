import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.bp = bp
        self.skin = skin
        self.insulin = insulin
        self.bmi = bmi
        self.dpf = dpf
        self.age = age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Pregnancies": [self.pregnancies],
                "Glucose": [self.glucose],
                "BloodPressure": [self.bp],
                "SkinThickness": [self.skin],
                "Insulin": [self.insulin],
                "BMI": [self.bmi],
                "DiabetesPedigreeFunction": [self.dpf],
                "Age": [self.age],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
