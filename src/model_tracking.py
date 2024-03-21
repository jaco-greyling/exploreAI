import mlflow
from mlflow.models.signature import infer_signature


class ModelTracker:
    def __init__(self, experiment_name: str, artifact_path: str, registered_model_name: str):
        """
        Initialize the ModelTracker class.

        Args:
            experiment_name (str): Name of the MLflow experiment.
            artifact_path (str): Path where artifacts will be stored.
            registered_model_name (str): Name of the model to be registered in MLflow.
        """
        self.experiment_name = experiment_name
        self.artifact_path = artifact_path
        self.registered_model_name = registered_model_name

    def setup(self, training_info: str = "Basic model training"):
        """
        Setup the MLflow experiment.

        Args:
            training_info (str): Description of the training process.
        """
        mlflow.set_experiment(self.experiment_name)
        mlflow.set_tag("Training Info", training_info)

    def run_experiment(self, params: dict, accuracy: float, X_train, lr):
        """
        Run the MLflow experiment.

        Args:
            params (dict): Hyperparameters for the model.
            accuracy (float): Accuracy of the model.
            X_train (numpy.ndarray): Training data.
            lr (sklearn.linear_model._Base): Trained model.
        """
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            signature = infer_signature(X_train, lr.predict(X_train))
            mlflow.log_model(
                model=lr,  # 'lr' is your trained model, which can be of any type
                artifact_path=self.artifact_path,
                registered_model_name=self.registered_model_name,
                signature=signature,
                input_example=X_train[:1],
            )
