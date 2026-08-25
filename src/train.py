from data_prep import LoadData, Data_prep, dataSplite,scaled_Data
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,f1_score,recall_score,
    precision_score,confusion_matrix)


logistic_pipeline = Pipeline([("scaler", StandardScaler()),("model", LogisticRegression())])
knn_pipeline = Pipeline([("scaler", StandardScaler()),("model", KNeighborsClassifier())])

# def initialize_Scaled_data(testSize,randomState):
#     here = Path(__file__).resolve()
#     repo_root = here.parent.parent
#     df = LoadData(repo_root / "data" / "creditcard.csv")  
#     X,y = Data_prep (df,'class')
#     X_train_Scaled,X_test_Scaled,y_train,y_test = scaled_Data(X,y,0.2,42)
#     return X_train_Scaled,X_test_Scaled,y_train,y_test
def initialize_data(testSize,randomState):
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    df = LoadData(repo_root / "data" / "creditcard.csv")  
    X,y = Data_prep (df,'class')
    X_train,X_test,y_train,y_test = dataSplite(X,y,testSize,randomState)
    return X_train,X_test,y_train,y_test


testSize = 0.2
randomState = 42


X_train,X_test,y_train,y_test = initialize_data(testSize,randomState)
LogisticR = Pipeline([
        ("scaler", StandardScaler()),
     ("model", LogisticRegression(max_iter=10000))
     ])
Knn = Pipeline([
    ("KNNClasifier",StandardScaler()),
    ("model",KNeighborsClassifier(n_neighbors=5))])
DecisionTree = DecisionTreeClassifier(random_state = randomState)
models = {
    "Logistic Regression": LogisticR,
    "KNN": Knn,
    "Decision Tree": DecisionTree
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring =  {
            "accuracy" : "accuracy",
            "precision" : "precision",
            "recall" : "recall",
            "f1" : "f1",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "balanced_accuracy": "balanced_accuracy"
        }
metrics = ['Mean accuracy','Mean Precision','Mean Recall','Mean F1','Mean roc_auc','Mean average_precision','Mean balanced_accuracy']
Report_results = {'0':['Model'],
                    '1' :['accuracy'] ,
                    '2' :['Mean Precision'],
                    '3' :['Mean Recall'], 
                    '4' :['Mean F1'] ,
                    '5' :['Mean roc_auc'],
                    '6' :['Mean average_precision'],
                    '7' :['Mean balanced_accuracy']}
for name, model in models.items():
    Cross_validation_Result = cross_validate(
        model,
        X_train,
        y_train,
        cv = skf,
        scoring = scoring,
        return_train_score=True
    )

    Report_results['0'].append(name)
    i = 1
    for metric in scoring.keys():
        mean_score = Cross_validation_Result[f'test_{metric}'].mean()
        Report_results[str(i)].append(mean_score)
        i = i+1

Report_results = pd.DataFrame(Report_results)
print(Report_results)

 ############################

 ####### Gride Search

param_grids = {
    "Logistic Regression": {
        'max_iter' : [100,1000,5000,10000]
    },
    "KNN": {
        "k": [1, 5 ,20]
    },
    "Decision Tree": {
        "max_depth": [2, 5, 10, None]
    }
}
    # grid.fit(X_train, y_train)

    # Report_results[name] = grid

    










 

