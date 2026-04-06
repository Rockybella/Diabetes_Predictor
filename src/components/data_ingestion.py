import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

# Simple config to make logs show up in your terminal
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)


@dataclass
class DataIngestionConfig:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ["Preg", "Plas", "Pres", "Skin", "Test", "BMI", "Pedigree", "Age", "Class"]

    X_train_data_path: str = os.path.join("artifacts", "X_train.csv")
    X_test_data_path: str = os.path.join("artifacts", "X_test.csv")
    y_test_data_path: str = os.path.join("artifacts", "y_test.csv")
    y_train_data_path: str = os.path.join("artifacts", "y_train.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # 1. Read the dataset
            # Ensure 'diabetes.csv' is in your 'notebook' folder
            df = pd.read_csv(
                self.ingestion_config.url, names=self.ingestion_config.names
            )
            logging.info("Read the dataset as dataframe")

            # 2. Data Cleaning
            # Replacing 0 with Median for specific columns
            cols_to_fix = [
                "Preg",
                "Plas",
                "Pres",
                "Skin",
                "Test",
                "BMI",
                "Pedigree",
                "Age",
                "Class",
            ]
            # Note: Your notebook used different names (Plas, Pres, etc).
            # I used standard Pima names here; change them if your CSV headers differ!

            logging.info("Replacing zero values with median")
            for col in cols_to_fix:
                df[col] = df[col].replace(0, df[col].median())

            # 3. Save Raw Data
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True,
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # 4. Train-Test Split
            logging.info("Train test split initiated")
            # train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            # --- SAMPLING CONFIGURATION ---
            # Options: 'linear', 'shuffled', 'stratified', 'automatic'
            sampling_method = "automatic"

            X = df.drop("Class", axis=1)
            y = df["Class"]

            if sampling_method == "linear":
                # Linear: Divides partitions without changing sequence
                split_idx = int(len(df) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            elif sampling_method == "shuffled":
                # Shuffled: Split randomly/arbitrarily
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=True
                )

            elif sampling_method == "stratified" or sampling_method == "automatic":
                # Stratified/Automatic: Ensures static class distribution
                # Defaults to stratified because 'Class' is a binomial feature
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

            X_train.to_csv(
                self.ingestion_config.X_train_data_path, index=False, header=True
            )
            y_train.to_csv(
                self.ingestion_config.y_train_data_path, index=False, header=True
            )
            X_test.to_csv(
                self.ingestion_config.X_test_data_path, index=False, header=True
            )
            y_test.to_csv(
                self.ingestion_config.y_test_data_path, index=False, header=True
            )

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.X_train_data_path,
                self.ingestion_config.X_test_data_path,
                self.ingestion_config.y_train_data_path,
                self.ingestion_config.y_test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # 1. Create the object
    obj = DataIngestion()

    # 2. Start the process
    # This calls your initiate_data_ingestion function
    X_train_data, X_test_data, y_train_data, y_test_data = obj.initiate_data_ingestion()

    # 3. Add a print to prove it worked
    print("Ingestion completed successfully!")
