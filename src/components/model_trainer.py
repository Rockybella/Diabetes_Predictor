import os
import sys
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    # Adding type hint for dataclass standard
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        # Define the L1 regularization value
        self.l1_val = 0.000010

    def create_nn_model(self):
        """Creates the Neural Network based on your architecture."""
        try:
            model = models.Sequential(
                [
                    layers.Input(shape=(8,)),
                    layers.Dense(
                        50,
                        activation="relu",
                        kernel_regularizer=regularizers.l1(self.l1_val),
                        name="Layer_2",
                    ),
                    layers.Dense(
                        50,
                        activation="relu",
                        kernel_regularizer=regularizers.l1(self.l1_val),
                        name="Layer_3",
                    ),
                    layers.Dense(
                        2,
                        activation="softmax",
                        kernel_regularizer=regularizers.l1(self.l1_val),
                        name="Layer_4",
                    ),
                ]
            )

            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(
                "Splitting training and test input data from transformation arrays"
            )
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # --- 1. Train Neural Network ---
            logging.info("Building and Training Neural Network")
            nn_model = self.create_nn_model()
            nn_model.summary()  # This works here because it's a Keras model
            nn_model.fit(
                X_train, y_train, epochs=10, verbose=0
            )  # Adjust epochs as needed

            # --- 2. Train Voting Classifier ---
            models_dict = {
                "SVM": SVC(probability=True, kernel="rbf", C=10, gamma=0.1),
                "DT": DecisionTreeClassifier(
                    criterion="entropy", max_depth=10, random_state=42
                ),
                "NB": GaussianNB(),
            }

            ensemble = VotingClassifier(
                estimators=[
                    ("svm", models_dict["SVM"]),
                    ("dt", models_dict["DT"]),
                    ("nb", models_dict["NB"]),
                ],
                voting="soft",
            )

            logging.info("Training the hybrid ensemble model")
            ensemble.fit(X_train, y_train)

            # --- 3. Save and Evaluate ---
            # Saving the Ensemble as the primary model for deployment
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=ensemble,
            )

            predicted = ensemble.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)

            logging.info(f"Model training completed with accuracy: {acc_score}")
            return acc_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation

        logging.info("Starting the full training pipeline")

        # 1. Ingest Data - Capture all 5 paths returned by your script
        ingestion = DataIngestion()
        X_train_path, X_test_path, y_train_path, y_test_path, raw_path = (
            ingestion.initiate_data_ingestion()
        )

        # 2. Transform Data - Pass the 4 specific split paths
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            X_train_path=X_train_path,
            X_test_path=X_test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path,
            raw_data_path=raw_data_patch,
        )

        # 3. Train Model
        trainer = ModelTrainer()
        accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"Pipeline Completed! Final Model Accuracy: {accuracy}")

    except Exception as e:
        raise CustomException(e, sys)
