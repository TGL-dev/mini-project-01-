## 1. Introduction 
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

| Model               |     Accuracy |    Precision |       Recall |           F1 |
| :------------------ | -----------: | -----------: | -----------: | -----------: |
| Logistic Regression |     0.999169 |     0.857424 |     0.587042 |     0.695177 |
| *KNN*               |   0.999510   |   0.923021   |   0.763717   |   0.833982   |
| Decision Tree       |     0.999086 |     0.717067 |     0.731063 |     0.722624 |

| Model               |      ROC-AUC | Average Precision | Balanced Accuracy |
| :------------------ | -----------: | ----------------: | ----------------: |
| Logistic Regression |   0.974604   |          0.720886 |          0.793441 |
| *KNN*               |     0.907522 |      0.771939     |        0.881805   |
| Decision Tree       |     0.865292 |          0.523955 |          0.865292 |



*False Negative costs componies money so in this project and dataset to find fraudukant transactions, we use Recall and F1 to choose the best model. we can also use roc_auc because of our unblanced data.* 
*ROC curve reflects the test's ability to distinguish *between fraud and normal transaction. AUC values range from 0.5 to 1.0, *with a value of 0.5 indicating that the test is no better than chance at *distinguishing between fraudulant and normal transaction*
*combining Recall,F1 and roc_auc together as metrics to chose the best model I think that KNN is the best model among these 3 models use for detecting fradulant transactions* 

## 5. Scaling Experiment
# Mandatory Experiment 1: Effect of Scaling

results for KNN :

| Model | Scaling           | Precision | Recall    | F1        |
|-------|-------------------|-----------|-----------|-----------|
| KNN   | Without Scaling   |    0.6    | 0.010885  | 0.021333  |
| KNN   | With Scaling      | 0.923021  | 0.763717  | 0.833982  | 


# Explaining:
using KNN without scaling the data coused drastic change in recall and F1 and made the model useless making it unable to predict the real results!

- Why is KNN sensitive to scaling? KNN uses distance to find neighbors.feature with bigge scale will effect more on KNN calculation process. features with small scale will be almost ignored due to their short distance as neghbors.
- Why is Decision Tree less sensitive? Decision Tree doesn't use distance in it's process. it uses one feature at a time to proceed its training so other features and their scale won't matter.

---

## 6. Hyperparameter Experiment

`Max_depth` Hyperparameter for Decision Tree

| max_depth   |Train Precision  |CV Test Precision | Train Recall |CV Test Recall|Train F1  |CV Test F1|
|:-----------:|----------------:|-----------------:|-------------:|-------------:|---------:|------:|
| 2           | 0.839           | 0.815            | 0.716        | 0.698        | 0.772    | 0.750 |
| *5*         | 0.951           | 0.903            | 0.792        | 0.731        | 0.864    | 0.806 |
| 10          | 1.000           | 0.848            | 0.850        | 0.709        | 0.919    | 0.772 |
| None        | 1.000           | 0.724            | 1.000        | 0.742        | 1.000    | 0.732 |

as you can see somwhere between max_depth 0f 5 and 10  overfitting has occured! 
so here the best hyperparameter for decision Tree is :max_depth = 5 

`n_neighbors`  Hyperparameter for KNN

| `n_neighbors` |CVTest Precision| Train Precision |CVTest Recall| Train Recall |CVTest F1| Train F1 |
|:-------------:|---------------:|----------------:|------------:|-------------:|--------:|---------:|
| 1             | 0.850397       | 1.000000        | 0.771751    | 1.000000     | 0.806235| 1.000000 |
| 5             | 0.923021       | 0.940874        | 0.763717    | 0.777180     | 0.833982| 0.851119 |
| 20            | 0.874053       | 0.877078        | 0.714661    | 0.722133     | 0.785680| 0.791954 |

as you can see somwhere between max_depth 0f 5 and 1  overfitting has occured! 
so here the best hyperparameter for KNN is : n_neighbors = 5 


## 7. Impact of Classification Threshold

Most classification models use a default decision threshold of 0.5.

```thresholds
0.3
0.5
0.7
```
chosen model for this experiment was KNN

| Threshold | Precision | Recall   | F1       | TN      | FP  | FN   | TP   |
|----------:|----------:|----------|----------|---------|-----|------|------|
| 0.3       | 0.899458  | 0.790892 | 0.839777 | 45192.2 | 6.6 | 15.4 | 58.2 |
| 0.5       | 0.923021  | 0.763717 | 0.833982 | 45194.0 | 4.8 | 17.4 | 56.2 |
| 0.7       | 0.944812  | 0.679341 | 0.789502 | 45195.8 | 3.0 | 23.6 | 50.0 |

- What happens to Recall when the threshold decreases?
Recall decreases because less transactions are classified as fraud also we have less fals positive as the thershold increases
- What happens to Precision?
precision increases but looking at the TP and FP you see both of them were decreasing so we can't say that because the precision got better we have a better model.
- Which threshold would you recommend for a fraud detection system?
thereshold = 0.3
- What trade-off does your chosen threshold create?
with the threshold of 0.3 the model can detect more fraud but for more false positive meaning it will catch more normal transaton as fraud too but the price is indurable indeed!

## 8. Final Model Selection
I have chosen KNN with 5 n_neghbor as the best model for this dataset I have also saved the decision tree model with haperparameter of max_depth = 5 to compare the results on test data.the results are described below:

| Model         | Accuracy | Recall   | Precision| TN    | FP | FN | TP |
|:--------------|----------|----------|----------|-------|---:|---:|---:|
| KNN           | 0.999470 | 0.771739 | 0.887500 | 56490 | 9  | 21 | 71 |
| Decision Tree | 0.999541 | 0.782609 | 0.923077 | 56493 | 6  | 20 | 72 |

the results are so close! it seems that decision Tree can predict fraud and normal better with less false positive and false negetive!
since I used threshold = 0.5 for KNN i think that the prediction could have been better if I used threshold = 0.3.
I will update the result for threshold = 0.3 !

update!
| Model         | Accuracy | Recall   | Precision | TN    | FP  | FN | TP |
|:--------------|----------|----------|-----------|-------|-----|---:|---:|
| KNN           | 0.999487 | 0.815217 | 0.862068  | 56487 | 12  | 17 | 75 |
| Decision Tree | 0.999540 | 0.782608 | 0.923076  | 56493 | 6   | 20 | 72 |

As you can see KNN with threshold of 0.3 catched more TP with less FN making it better than decision Tree model in camparison!



### Guiding Questions

1. Which model do you expect to perform best for fraud detection? Why?
Decision Tree : I though this Desision Tree can model unblanced dataset and it can model nonlinearity better.

2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?
Recall because it is important that we don't miss any Fradulant transation since it cost companie money but we also want to have good precision meaning that we need to have high TP count so we wouldn't be in trouble for checking False positive since F1 has FN and Tp both so F1 and recall are both important 

3. What do you expect to happen if the model predicts all transactions as legitimate?
the model is too simple and is unable to find the connection between features and target 

4. Do you expect feature scaling to significantly affect KNN performance?
yes. since distance between the point we are at in KNN and diffrent features it will have significant 
impact on the output since diffrent features are in diffrent ِNumerical Range.  features in higher number range have lower effect on the output than those with lower number range .

5. Do you expect the Decision Tree to overfit? Why?
if max_depth is not limited or the number of leaf , yes overfitting can happen!


### After Training Analysis

Was your initial hypothesis correct? 
No!
Which model performed best?
KNN
Which metric was most informative?
f1 and recall
How did class imbalance affect the results?

What was the trade-off between False Positives and False Negatives?
