from data_prep import LoadData, Data_prep, dataSplite
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score
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
        "KNN": joblib.load(modelPath /"KNN.pkl"),
        "Decision Tree": joblib.load(modelPath /"DecisionTree.pkl")
    }
    return models

def SaveOutputResult(results,output_path):
    with open(output_path,"w",encoding="utf-8") as file:
        json.dump(results,file,indent=4,ensure_ascii=False)

def PredictModels(models, X_test, y_test, path):

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        print(f"results metrics for model {name} are descibed below :")
        print(f"Accuracy : {accuracy}")
        print(f"Recall   : {recall}")
        print(f"Precision: {precision}")
        print("Confusion Matrix:")
        print("------------------------------------------------------")
        print(cm)
        results = []
        for prediction, probability ,actual in zip (predictions,probabilities,y_test):
            if prediction == 1:
                prediction_name = "Fraud"
            else:
                prediction_name = "Normal"
            if prediction == actual:
                OutputStatus = "success"
            else:
                OutputStatus = "failed"
            result = {
                "prediction": prediction_name,
                "class_id": int(prediction),
                "probability": probability,
                "status": OutputStatus
            }
            results.append(result)
        output_path = path / f"output_{name}.json"
        SaveOutputResult(results,output_path)


if __name__ == "__main__":
    data_path = "G:/AI_Course/mini-project-01/data/Test_data.csv"
    X_test,y_test = LoadTestData(data_path)
    model_path = "G:/AI_Course/mini-project-01/models"
    models = load_models(model_path)
    outputPath = "G:/AI_Course/mini-project-01/reports"
    PredictModels(models,X_test,y_test,outputPath)


