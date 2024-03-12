import joblib


class ModelPersistor:
    @staticmethod
    def save_model(model, filepath):
        """Saves model to disk.

        Args:
            model: ml model.
            filepath (_type_): path to save the model.
        """
        try:
            joblib.dump(model, filepath)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    @staticmethod
    def load_model(filepath):
        """Loads model from disk.

        Args:
            filepath (str): path to the model file
        """
        try:
            model = joblib.load(filepath)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
