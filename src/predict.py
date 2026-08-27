from data_prep import LoadData, Data_prep, dataSplite
import pandas as pd
import joblib
import json
from pathlib import Path

import pandas as pd
import joblib
import json

from pathlib import Path


def LoadTestData(path):

    test_data = pd.read_json(
        path,
        orient="records"
    )

    X_test = test_data.drop(columns=["class"])
    y_test = test_data["class"]
    return X_test, y_test

def load_models(path):

    modelPath = Path(path)
    models = {
        "KNN": joblib.load(modelPath / "KNN.pkl"),
        "Decision Tree": joblib.load(path / "DecisionTree.pkl")
    }
    return models


if __name__ == "__main__":
    data_path = "G:/AI_Course/mini-project-01/data/Test_data.csv"
    LoadTestData(data_path)
    model_path = "G:/AI_Course/mini-project-01/models/"
    load_models()
    