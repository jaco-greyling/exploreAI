import mlflow
from loguru import logger

from src.model_tracking import ModelTracker
from src.models.explainable_boosting_machine import EBM
from src.pre_processor.pre_processor import PreProcessor

ebm_params = {
    "max_bins": 512,
    "max_interaction_bins": 16,
    "interactions": 10,
    "outer_bags": 4,
    "inner_bags": 0,
    "learning_rate": 0.001,
    "min_samples_leaf": 2,
    "max_leaves": 3,
    "early_stopping_rounds": 100,
    "n_jobs": -1,
    "random_state": 42,
}


class ModelTraining:
    def __init__(self, data: str):
        """
        Initialize the ModelTraining pipeline.

        Args:
            data (str): Path to the training data.
        """
        self.models = ["EBM"]
        self.data = data

    def run(self) -> None:
        """
        Run the ModelTraining pipeline.

        This function trains multiple models and chooses the best one based on evaluation metrics.
        """
        # Initialize the PreProcessor and ModelTracker classes
        logger.info("Initializing PreProcessor and ModelTracker classes.")
        preprocessor = PreProcessor(production=False)
        X_train, X_test, y_train, y_test = preprocessor.transform(self.data)

        model_tracker = ModelTracker(
            experiment_name="First test experiment", artifact_path="logged_models", registered_model_name="EBM"
        )
        model_tracker.setup(training_info="EBM model training")

        logger.info("Starting MLflow experiment")
        if mlflow.active_run():
            mlflow.end_run()

        for model_name in self.models:
            logger.info("Training models")
            if model_name == "EBM":
                model = EBM()
                model.fit(X_train, y_train)
                logger.info(f"{model_name} model trained. Logging metrics to MLflow.")
                # It's assumed that run_experiment will calculate and log the accuracy internally
                model_tracker.run_experiment(
                    params=ebm_params, model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
                )
            else:
                raise ValueError(f"Model {model_name} not supported.")
