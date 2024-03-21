import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score


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

    def run_experiment(self, params: dict, model, X_train, y_train, X_test, y_test):
        """
        Run the MLflow experiment.

        Args:
            params (dict): Hyperparameters for the model.
            model: Trained model.
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            X_test (numpy.ndarray): Testing data features.
            y_test (numpy.ndarray): Testing data labels.
        """
        with mlflow.start_run():
            # Log model parameters
            mlflow.log_params(params)

            # Evaluate the model
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, train_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)

            # Log accuracy metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)

            # Log the model
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.pyfunc.log_model(
                artifact_path=self.artifact_path,
                python_model=model,
                registered_model_name=self.registered_model_name,
                signature=signature,
            )
