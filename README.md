
* Business scenario
credit card companies must be able to recognize fraudulent credit card transactions so that customers would'nt be charged for items they did not purchase. After all, customers will not be willing to provide their credit card details unless they are assured regarding the security of their information and the detection of financial transactions not initiated by them.
* Objective
The goal is to develop a model to predict and detect fraudulent activities in credit card transactions.
* Dataset
To achieve the goal the dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud will be used that contians transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where there is 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.


## 2. Data Analysis

* Dataset statistics
There are 284807 raw Samples in this dataset 
* Feature information
this dataset contains 30 feature including Time , Amount of transaction and 28 other features (V1,V2,...V28). the origin of these 28 features(V1 ... V28) is not specified because of the security reasons. since all these features were float there was no need to seprate sting and numberic data and change the string feature into numberic values.
* Missing value analysis
there were zero missing values across the dataset
there are 1081 duplicate rows in the dataset witch were removed

## 3. Initial Hypothesis
for these  two reasons below I think that  decission Tree Classifier can have a really good ability to predict the fraud transactions. 
1 - we have unbalanced dataset and the labels are seperated into two classes: 0 and 1
2- I believe the relation between features and label are nonlinear
KNN clasifier can be the best next model. 

## 4. Model Comparison


* Models and Metrics with Cross-validation results

0         1               2            3         4  \
0                Model  accuracy  Mean Precision  Mean Recall   Mean F1   
1  Logistic Regression  0.999169        0.857424     0.587042  0.695177   
2                  KNN   0.99951        0.923021     0.763717  0.833982   
3        Decision Tree  0.999086        0.717067     0.731063  0.722624   

              5                       6                       7  
0  Mean roc_auc  Mean average_precision  Mean balanced_accuracy  
1      0.974604                0.720886                0.793441  
2      0.907522                0.771939                0.881805  
3      0.865292                0.523955                0.865292

*False Negative costs componies money so in this project and dataset to find fraudukant transactions, we use Recall and F1 to choose the best model. we can also use roc_auc because of our unblanced data.* 
*ROC curve reflects the test's ability to distinguish *between fraud and normal transaction. AUC values range from 0.5 to 1.0, *with a value of 0.5 indicating that the test is no better than chance at *distinguishing between fraudulant and normal transaction*
*combining Recall,F1 and roc_auc together as metrics to chose the best model I think that KNN is the best model among these 3 models use for detecting fradulant transactions* 



## 5. Scaling Experiment

Explain the effect of scaling.

## 6. Hyperparameter Experiment

Explain the impact of parameter changes.

## 7. Final Model Selection

Explain why your final model was selected.

## 8. Running Instructions

Explain:

* Installation
* Training
* Prediction

## 9. Reflection

Answer the questions below.

---


# 7. Hypothesis Before Modeling

Before training your models, write your hypotheses in `README.md`.

### Guiding Questions

1. Which model do you expect to perform best for fraud detection? Why?
Decision Tree : I though this Desision Tree can model unblanced dataset and it can model nonlinearity better.

2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?
Recall because it is important that we don't miss any Fradulant transation since it cost companie money but we also want to have good precision meaning that we need to have high TP count so we wouldn't be in trouble for checking False positive since F1 has FN and Tp both so F1 and recall are both important 

3. What do you expect to happen if the model predicts all transactions as legitimate?

4. Do you expect feature scaling to significantly affect KNN performance?

5. Do you expect the Decision Tree to overfit? Why?

### Example Hypothesis

> I expect Logistic Regression to provide a strong baseline because the problem is binary classification. However, I expect the Decision Tree to capture nonlinear relationships that Logistic Regression may miss.

> I expect Recall to be particularly important because failing to detect a fraudulent transaction can have a significant cost. However, maximizing Recall alone may produce too many false alarms, so Precision and F1-score should also be considered.

### After Training Analysis

Explain in your `README.md`:

- Was your initial hypothesis correct?
- Which model performed best?
- Which metric was most informative?
- How did class imbalance affect the results?
- What was the trade-off between False Positives and False Negatives?