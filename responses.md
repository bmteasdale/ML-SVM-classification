# Machine Learning (CSCI 444)  Assignment \#2  
## Question \#1:  
|     |   svm_linear |   svm_rbf |       rf |     knn1 |     knn5 |    knn10 |
|:----|-------------:|----------:|---------:|---------:|---------:|---------:|
| err |     0.028109 |  0.008859 | 0.016184 | 0.011244 | 0.012606 | 0.013799 |  

#### a) Which classifier performs the best in this task?  
By using the error rates given in the programs output, we can see that the SVM with the 'rbf' kernel performed the best for this task with an error rate of only 0.008859 (99.1141% accurate)
#### b) Why do you think this classifier outperforms others?  
We can use the fact that the SVM model using rbf kernel performed better to can conclude that the decision boundary between the two classes in our problem (8's and 9's) must be non-linear, hence a non-linear kernel would do a better job for separation of the binary classes (8 or 9).
#### c) How does KNN compare to the results obtained in assignment 1? Why do you observe this comparative pattern?  
KNN classifier with K=1 had an error rate of 0.011244, K=5 had an error rate of 0.012606, K=10 had an error rate of 0.013799, and because of this we can see that as K increased, our error rate had also increased. In assignment 1, the KNN classifier had this same trait but the KNN, at very low K values, was remaining at 0 for an error rate until K grew larger. Although the KNN classifier error rates still are low in this assignment, they differ for the reason that the rise in error rates is not as drastic as K values increase.

## Question \#2:  
My dataset consists of data collected from:
[Google Toolbox Datasets](https://datasetsearch.research.google.com/search?query=Breast%20Cancer%20Wisconsin%20(Diagnostic)%20Data%20Set&docid=1XGK7COpYqrz3u9VAAAAAA%3D%3D "Link to Breast Cancer Dataset")

#### Number of samples
```python
# of samples in dataset: 569
```

#### Number of Measurements
```python
# of measurements in dataset: 32
```
#### Group of Interest:
The group of interest in my dataset is the feature "Diagnosis". The value of Diagnosis is binary, B for Benign and M
for Malignant. I later encoded these values to be either 0 or 1 so that all data remained numerical. Diagnosis is
what my program in Question 3 attempts to classify when inputted many measurements.

#### Sample Count for Group of Interest (Malignant):
```python
# Count of Malignant (Group of Interest) Samples: 212
```
#### Sample Count for Group not of Interest (Benign):
```python
# Count of Benign (Group not of Interest) Samples: 357
```
#### Collecting the AUC's:  
| Features             |    AUC |
|:---------------------|-------:|
| perimeter_worst      | 0.0245 |
| radius_worst         | 0.0296 |
| area_worst           | 0.0302 |
| concave points_worst | 0.0333 |
| concave points_mean  | 0.0356 |
| perimeter_mean       | 0.0531 |
| area_mean            | 0.0617 |
| concavity_mean       | 0.0622 |
| radius_mean          | 0.0625 |
| area_se              | 0.0736 |
## Question \#3:  
|     |   svm_linear |   svm_rbf |       rf |     knn1 |     knn5 |   knn10 |
|:----|-------------:|----------:|---------:|---------:|---------:|--------:|
| err |     0.024592 |  0.024608 | 0.040382 | 0.043922 | 0.035119 | 0.03338 |
#### Question: Is the best performing classifier from Question 1 the same in Question 3? Elaborate on those similarities/differences – what about your dataset may have contributed to the differences/similarities observed?
The best performing classifier in this case was SVM with the linear kernel but it wasn't a winner by much. svm_linear had an error rate of 0.024592 (97.54% accurate) while svm_rbf had an error rate of 0.024608 leaving a difference of only 0.000016. Due to the fact that the linear kernel outperformed random forest, and all knn classifiers, we can conclude that the decision boundary between the two binary classes (Benign or Malignant diagnoses) is more linear when compared to the boundary line between the hand written digits classes (8 or 9).
