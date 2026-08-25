#### pre prossesing Data
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def LoadData(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape : {df.shape}")
    print('--------------------------------')
    print(f"Dataset Information :{df.info()}")
    print('--------------------------------')
    print(df.head())
    print('--------------------------------')
    print(f"Dataset type :{df.dtypes}")
    return df

def Data_prep(df,target):
    print(f" how many mising value each column in the dataset have?\n{df.isnull().sum()}")
    print("Duplicate rows:", df.duplicated().sum())
    df_cleaned = df.drop_duplicates(keep=False)
    print("Duplicate rows:", df_cleaned.duplicated().sum())
    X = df_cleaned.drop("Class",axis=1)
    y = df_cleaned["Class"]
    print ("X: ", np.shape(X))
    print ("y: ", np.shape(y))
    print("Number of class 0 and 1 :")
    print(y.value_counts())     # Target 0 and 1 count
    print("class 0 and 1's distrbution is as below :(percent)")
    print(y.value_counts(normalize=True)*100)   # Target 0 and 1 percentage
    df_cleaned.describe()
    fig, axs = plt.subplots(len(df_cleaned.columns), 1, figsize=(7, 18), dpi=95)
    for i, col in enumerate(df_cleaned.columns):
        axs[i].boxplot(df_cleaned[col], vert=False)
        axs[i].set_ylabel(col)
    plt.tight_layout()
    plt.show()
    return X,y
def dataSplite(X,y,testSize,randomState):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = testSize,
        random_state = randomState,
        stratify = y
    )
    return X_train,X_test,y_train,y_test
def scaled_Data(X,y,testSize,randomState):
    X_train,X_test,y_train,y_test = dataSplite(X,y,testSize,randomState)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test,y_train,y_test



# df = LoadData('G:/AI_Course/mini-project-01/data/creditcard.csv')
# X,y = Data_prep(df,'class')
# X_train,X_test,y_train,y_test = scaled_Data(X,y,0.2,42)
