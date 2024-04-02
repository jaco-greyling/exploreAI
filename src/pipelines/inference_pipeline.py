import pandas as pd

from src.model_persistor import ModelPersistor
from src.pre_processor.pre_processor import PreProcessor


class ModelInference:
    """
    Class for performing model inference.
    """

    def __init__(self, model_path: str):
        """
        Initialize the ModelInference class.

        Args:
            model_path (str): Path to the pickle file containing the trained model.
        """
        self.model_path = model_path
        self.model = None
        self.model_persistor = ModelPersistor()

    def run(self, data: list) -> list:
        """
        Perform model inference on the given data.

        Args:
            data (list): Input data for prediction.

        Returns:
            list: Predicted output.
        """
        self.model = self.model_persistor.load_model()

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        preprocessor = PreProcessor(production=True)
        X, y = preprocessor.transform(data)
        # Perform inference using the loaded model
        predictions = self.model.predict(pd.DataFrame(X))

        return predictions
