# Machine Learning Project


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.4
```
## Read In Data

Here, I read in the training dataset to use in modeling.  I am also going to check what the dataset looks like and see what the Classe variable distribution is.


```r
pml.training <- read.csv("~/Downloads/pml-training.csv")
dim(pml.training)
```

```
## [1] 19622   160
```

```r
# names(pml.training) - commented out to save space
table(pml.training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
## Clean data

Now, I am going to remove the items that have a lot of missing/NA data points and name it my.training This will allow the analysis to run and also make my data files more manageable.  I found the missing ones by looking at the dataset.


```r
my.training <- pml.training[,c(6:11,37:49,60:68,84:86,102,113:124,140,151:160)]
```
## Prepare training data by spliting it up into a Train and Test

Split up Training Set into a Training and Testing

```r
set.seed(1432)
inTrain = createDataPartition(my.training$classe, p=.70)[[1]]
training = my.training[inTrain,]
testing = my.training[-inTrain,]
dim(training); dim(testing)
```

```
## [1] 13737    55
```

```
## [1] 5885   55
```
## Choosing a model to use
I first tried CART analysis on this but it didn't do a very good job so I left the output off this .Rmd document.

I heard that RandomForest is good at predicting so I am going to try to use the rf option in train (from the caret package).  Professors mentioned that Random Forest is used in many competitions so I thought that would be a good fit for this since accuracy was key and explaining the exact model was not the main goal of this.


```r
modFitRF <- train(classe~.,methods="rf",data=training)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
print(modFitRF$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, methods = "rf") 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 28
## 
##         OOB estimate of  error rate: 0.23%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    0    0    0    1 0.0002560164
## B    8 2647    1    2    0 0.0041384500
## C    0    6 2390    0    0 0.0025041736
## D    0    0    7 2244    1 0.0035523979
## E    0    1    0    4 2520 0.0019801980
```
Now that we have the RandomForest model (modFitRF), we should check to see how it did at predicting the training and testing sets


```r
table(training$classe,predict(modFitRF,training))
```

```
##    
##        A    B    C    D    E
##   A 3906    0    0    0    0
##   B    0 2658    0    0    0
##   C    0    0 2396    0    0
##   D    0    0    0 2252    0
##   E    0    0    0    0 2525
```

```r
table(testing$classe, predict(modFitRF,testing))
```

```
##    
##        A    B    C    D    E
##   A 1673    1    0    0    0
##   B    1 1138    0    0    0
##   C    0    1 1025    0    0
##   D    0    0    4  960    0
##   E    0    0    0    6 1076
```

```r
predRF.tr <- predict(modFitRF,training)
accuracyRF.tr = sum(predRF.tr ==training$classe)/length(predRF.tr)
accuracyRF.tr
```

```
## [1] 1
```

```r
predRF.te <- predict(modFitRF,testing)
accuracyRF.te = sum(predRF.te ==testing$classe)/length(predRF.te)
accuracyRF.te
```

```
## [1] 0.997791
```


## Using Cross Validation with Random Forest model
The assignment asked to use Cross Validation so this is below using Random Forest with Cross Validation.  I am using a K=5 to add some accuracy but to not have it take forever on my computer.  That means we use 80% of the data each time so that seems reasonable to me. Plus, I already split out 70% into my training set so I have a set to test the error on after coming up with my model.


```r
ctrl <- trainControl(method = "cv", number  = 5)
modFitRF2 = train(classe~ . , data = training, method = 'rf', trControl = ctrl)
modFitRF2$results
```

```
##   mtry  Accuracy     Kappa   AccuracySD     KappaSD
## 1    2 0.9923564 0.9903308 0.0020751830 0.002625948
## 2   28 0.9972337 0.9965009 0.0009491615 0.001200754
## 3   54 0.9942492 0.9927250 0.0014189177 0.001795477
```
## Checking accuracy on the Test holdout set

I ran the model with Cross Validation and now I want to check the accuracy on my test set 

```r
predRF2 = predict(modFitRF2, testing)
table(predRF2,testing$classe)
```

```
##        
## predRF2    A    B    C    D    E
##       A 1673    1    0    0    0
##       B    1 1138    1    0    0
##       C    0    0 1025    5    0
##       D    0    0    0  959    6
##       E    0    0    0    0 1076
```

```r
accuracyRF2 = sum(predRF2 ==testing$classe)/length(predRF2)
accuracyRF2
```

```
## [1] 0.9976211
```
The accuracyRF2 above is my best guess at the out of sample error since it is based on a holdout sample I had in the original Training set. Since the model was not built using this data it should be a good representation of the out of sample error.

In my opinion, this Model does a great job of predicting the classe variable and I will use that for predicting the 20 Test sets.

## Predicting the 20 Test sets using the Random Forest model I developed above.

I first need to read in the Test data and I made it look like the test set I used by removing the variables that were NA and missing in the Training set.

```r
pml.testing <- read.csv("~/Downloads/pml-testing.csv")
pml.testing2 <- pml.testing[,c(6:11,37:49,60:68,84:86,102,113:124,140,151:160)]

dim(pml.testing2)
```

```
## [1] 20 55
```

Here are my predictions for the 20 records in the test set. These have been submitted to the prediction submissions.


```r
predRF2.test = predict(modFitRF2,pml.testing2)
predRF2.test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```






