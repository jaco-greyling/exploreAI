import joblib
import loguru
import mlflow
from mlflow.tracking import MlflowClient

l_host = "127.0.0.1"
l_port = "8080"
mlflow.set_tracking_uri(uri=f"http://{l_host}:{l_port}")


class ModelPersistor:
    def __init__(self) -> None:
        self.client = MlflowClient()

    @staticmethod
    def save_model(model, filepath):
        """Saves model to disk.

        Args:
            model: ml model.
            filepath (_type_): path to save the model.
        """
        try:
            joblib.dump(model, filepath)
            loguru.info("Model saved successfully.")
        except Exception as e:
            loguru.info(f"Error saving model: {str(e)}")

    def load_model(self):
        """Loads model from disk.

        Args:
            filepath (str): path to the model file
        """
        experiment_name = "First test experiment"
        experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        # Search runs in the experiment
        runs = self.client.search_runs(experiment_ids=[experiment_id])

        # Find the run with the highest accuracy
        best_run = None
        best_accuracy = -float("inf")
        for run in runs:
            accuracy = run.data.metrics.get("test_accuracy", -float("inf"))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run = run

        if best_run:
            # ignore [F841]
            # best_run_id = best_run.info.run_id
            # model_uri = f"mlflow-artifacts:/{best_run_id}/artifacts/logged_models"
            # TODO: fix loading from mlflow client
            # ignore: [E501]
            model = mlflow.pyfunc.load_model(
                "/Users/jaco-pro/Documents/projects/exploreAI/mlartifacts/"
                + "725985855510154612/ece46319e9a14b7990e960aee4198d1a/artifacts/logged_models/"
            )
            print(f"Loaded best model based on accuracy: {best_accuracy}")
        else:
            print("No runs found for the experiment.")
            model = None

        return model
