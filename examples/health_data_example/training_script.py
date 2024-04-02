import pandas as pd

from src.pipelines.training_pipeline import ModelTraining

if __name__ == "__main__":
    # Read in data
    data = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

    model_trainer = ModelTraining(data)
    model_trainer.run()
    import pdb

    pdb.set_trace()
