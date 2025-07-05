import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        # You could load objects here once if needed
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            logging.info("Loading preprocessor and model objects")
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            logging.info("Transforming input features")
            data_scaled = preprocessor.transform(features)

            logging.info("Making prediction")
            preds = model.predict(data_scaled)

            # Optional: clip or constrain predictions if needed
            # preds = np.clip(preds, 0, 100)

            logging.info(f"Predictions completed: {preds}")
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            logging.info(f"Custom data dictionary: {custom_data_input_dict}")
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Custom data DataFrame created with shape {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)
