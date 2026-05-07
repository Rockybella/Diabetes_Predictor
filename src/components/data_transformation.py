import sys
import os
from dataclasses import dataclass

# CORRECT
from src.components.data_ingestion import DataIngestionConfig


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
   
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        
        try:
            scaler = StandardScaler()

            return scaler

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, X_train_path, X_test_path, y_train_path, y_test_path
    ):
        try:

            ingestion_cfg = DataIngestionConfig()
            names = ingestion_cfg.names
            feature_names = names[:-1]  # The 8 features
            target_name = ["Class"]  # The 1 target

            X_train_df = pd.read_csv(X_train_path, names=feature_names, header=0)
            X_test_df = pd.read_csv(X_test_path, names=feature_names, header=0)
            y_train_df = pd.read_csv(y_train_path, names=target_name, header=0)
            y_test_df = pd.read_csv(y_test_path, names=target_name, header=0)

            logging.info("Successfully read X and y with correct column counts")

            # 4. Skip the .drop() step because Class isn't in X_train_df anyway
            input_feature_train_df = X_train_df
            target_feature_train_df = y_train_df["Class"]

            input_feature_test_df = X_test_df
            target_feature_test_df = y_test_df["Class"]

            preprocessing_obj = self.get_data_transformer_object()

            
            target_column_name = "Class"

            # Separate Features and Target
            # --- Updated Separation Logic ---
            # X_train_df only has features and y_train_df only has the target

            input_feature_train_df = X_train_df
            target_feature_train_df = y_train_df[target_column_name]

            input_feature_test_df = X_test_df
            target_feature_test_df = y_test_df[target_column_name]

            logging.info("Applying StandardScaler on training and testing dataframes.")

            # Fit on training data, Transform both
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and targets back into arrays for the Trainer
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(
                f"Saving preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}"
            )

            # Use your utils.py to save the scaler
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # 1. Initialize the configs
    ingestion_config = DataIngestionConfig()
    obj = DataTransformation()

    # 2. Use the paths directly from the ingestion_config
    train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(
        X_train_path=ingestion_config.X_train_data_path,
        X_test_path=ingestion_config.X_test_data_path,
        y_train_path=ingestion_config.y_train_data_path,
        y_test_path=ingestion_config.y_test_data_path,
    )

    print("Data Transformation completed successfully!")
