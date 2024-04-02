import pandas as pd

from src.pipelines.inference_pipeline import ModelInference

if __name__ == "__main__":
    test_data = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

    model_inference_pipeline = ModelInference(model_path="test-path")

    model_predictions = model_inference_pipeline.run(test_data)
    import pdb

    pdb.set_trace()
