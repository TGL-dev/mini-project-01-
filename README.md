
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

since we have unbalanced dataset as said before I think that decission Tree Classifier can have a really good ability to predict the fraud transactions. KNN clasifier can be the best next model.

## 4. Model Comparison

Include:

* Models
* Metrics
* Cross-validation results

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

2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?

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