# Movement quality prediction
EG  
January 29, 2016  
## Overview
In this report, we aim to build a predictive model of the quality of movement recorded by fitness devices. In particular, we build a random forest model predicting movement classification based on accelleration measurements on the belt, forearm, dumbbell, and arm. There are five movement classes (from http://groupware.les.inf.puc-rio.br/har):

1) A: movement exactly according to specification (correct form)

2) B: throwing elbows to front (mistake)

3) C: lifting dumbbell only half way (mistake)

4) D: lowering dumbell only half way (mistake)

5) E: throwing hips to the front (mistake)

We find that a random forest model with 10-fold cross-validation generates a lower out of sample error than a linear discriminant analysis with 10-fold cross-validation trained on the same dataset (4.5% error vs. 48.2% error, respectively). Our predictive algorithm correctly predicts 19/20 of the classes in the test set.

## Pre-processing
Load libraries necessary for analysis.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

Load in training and testing sets.

```r
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url1, "training.csv", method = "curl")

url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url2, "testing.csv", method = "curl")

training <- read.table("training.csv", sep = ",", header = T, as.is = T)
testing <- read.table("testing.csv", sep = ",", header = T, as.is = T)
```

Set seed for analyses.

```r
set.seed(2222)
```

Limit variables to those relating to accelaration on belt, forearm, arm, and dumbbell. Also include the outcome variable.

```r
accell <- grep("accel", colnames(training))
training_accel <- training[,c(accell, which(colnames(training) == "classe"))]
```

Determine whethere there are any variables with missing values that can be removed from the dataset. Variance of accelleration in all areas missing the majority of observations; remove those variables.

```r
summary(training_accel)
```

```
##  total_accel_belt var_total_accel_belt  accel_belt_x       accel_belt_y   
##  Min.   : 0.00    Min.   : 0.000       Min.   :-120.000   Min.   :-69.00  
##  1st Qu.: 3.00    1st Qu.: 0.100       1st Qu.: -21.000   1st Qu.:  3.00  
##  Median :17.00    Median : 0.200       Median : -15.000   Median : 35.00  
##  Mean   :11.31    Mean   : 0.926       Mean   :  -5.595   Mean   : 30.15  
##  3rd Qu.:18.00    3rd Qu.: 0.300       3rd Qu.:  -5.000   3rd Qu.: 61.00  
##  Max.   :29.00    Max.   :16.500       Max.   :  85.000   Max.   :164.00  
##                   NA's   :19216                                           
##   accel_belt_z     total_accel_arm var_accel_arm     accel_arm_x     
##  Min.   :-275.00   Min.   : 1.00   Min.   :  0.00   Min.   :-404.00  
##  1st Qu.:-162.00   1st Qu.:17.00   1st Qu.:  9.03   1st Qu.:-242.00  
##  Median :-152.00   Median :27.00   Median : 40.61   Median : -44.00  
##  Mean   : -72.59   Mean   :25.51   Mean   : 53.23   Mean   : -60.24  
##  3rd Qu.:  27.00   3rd Qu.:33.00   3rd Qu.: 75.62   3rd Qu.:  84.00  
##  Max.   : 105.00   Max.   :66.00   Max.   :331.70   Max.   : 437.00  
##                                    NA's   :19216                     
##   accel_arm_y      accel_arm_z      total_accel_dumbbell
##  Min.   :-318.0   Min.   :-636.00   Min.   : 0.00       
##  1st Qu.: -54.0   1st Qu.:-143.00   1st Qu.: 4.00       
##  Median :  14.0   Median : -47.00   Median :10.00       
##  Mean   :  32.6   Mean   : -71.25   Mean   :13.72       
##  3rd Qu.: 139.0   3rd Qu.:  23.00   3rd Qu.:19.00       
##  Max.   : 308.0   Max.   : 292.00   Max.   :58.00       
##                                                         
##  var_accel_dumbbell accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z 
##  Min.   :  0.000    Min.   :-419.00   Min.   :-189.00   Min.   :-334.00  
##  1st Qu.:  0.378    1st Qu.: -50.00   1st Qu.:  -8.00   1st Qu.:-142.00  
##  Median :  1.000    Median :  -8.00   Median :  41.50   Median :  -1.00  
##  Mean   :  4.388    Mean   : -28.62   Mean   :  52.63   Mean   : -38.32  
##  3rd Qu.:  3.434    3rd Qu.:  11.00   3rd Qu.: 111.00   3rd Qu.:  38.00  
##  Max.   :230.428    Max.   : 235.00   Max.   : 315.00   Max.   : 318.00  
##  NA's   :19216                                                           
##  total_accel_forearm var_accel_forearm accel_forearm_x   accel_forearm_y 
##  Min.   :  0.00      Min.   :  0.000   Min.   :-498.00   Min.   :-632.0  
##  1st Qu.: 29.00      1st Qu.:  6.759   1st Qu.:-178.00   1st Qu.:  57.0  
##  Median : 36.00      Median : 21.165   Median : -57.00   Median : 201.0  
##  Mean   : 34.72      Mean   : 33.502   Mean   : -61.65   Mean   : 163.7  
##  3rd Qu.: 41.00      3rd Qu.: 51.240   3rd Qu.:  76.00   3rd Qu.: 312.0  
##  Max.   :108.00      Max.   :172.606   Max.   : 477.00   Max.   : 923.0  
##                      NA's   :19216                                       
##  accel_forearm_z      classe         
##  Min.   :-446.00   Length:19622      
##  1st Qu.:-182.00   Class :character  
##  Median : -39.00   Mode  :character  
##  Mean   : -55.29                     
##  3rd Qu.:  26.00                     
##  Max.   : 291.00                     
## 
```

```r
varAccell <- grep("^var", colnames(training_accel))
training_accel <- training_accel[,-varAccell]
```

Convert the outcome variables to factors to prepare for model building.

```r
training_accel$classe = as.factor(training_accel$classe)
```

## Predictive model
Use random forest algorithm to model excercise data, due to its high accuracy. Due to high computational load, allow for parallel processing to speed the process.

Specify that 10-fold cross-validation should be used, include control parameters in train function.

```r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = T)

modFit <- train(classe ~., training_accel, method = "rf", trControl = fitControl)

stopCluster(cluster)
```

Based on the 10-fold cross-validation, evaluate the out of sample error. Accuracy across the folds is 95.5%, with 4.5% estimated out of sample error.

```r
confusionMatrix(modFit)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 27.6  0.6  0.3  0.3  0.0
##          B  0.1 18.0  0.4  0.0  0.2
##          C  0.4  0.5 16.6  0.6  0.1
##          D  0.3  0.1  0.1 15.3  0.1
##          E  0.0  0.1  0.0  0.1 17.9
```

```r
accuracy <- 27.6 + 18 + 16.7 + 15.3 + 17.9
error <- 100 - accuracy
error
```

```
## [1] 4.5
```

### Predictions
Match testing set to training set.

```r
accell_test <- grep("accel", colnames(testing))
testing_accel <- testing[,c(accell_test, which(colnames(testing) == "problem_id"))]
varAccell_test <- grep("^var", colnames(testing_accel))
testing_accel <- testing_accel[,-varAccell_test]
```

Use random forest model to predict movement classification in test set.

```r
pred <- predict(modFit, testing_accel)
testing_accel$classe <- pred
testing_accel[, 17:18]
```

```
##    problem_id classe
## 1           1      B
## 2           2      A
## 3           3      C
## 4           4      A
## 5           5      A
## 6           6      E
## 7           7      D
## 8           8      B
## 9           9      A
## 10         10      A
## 11         11      B
## 12         12      C
## 13         13      B
## 14         14      A
## 15         15      E
## 16         16      E
## 17         17      A
## 18         18      B
## 19         19      B
## 20         20      B
```

## Supplemental analysis
Use linear discriminant analysis for the same set of data to confirm random forest is a good choice for the analysis.

```r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = T)

modFit2 <- train(classe ~., training_accel, method = "lda", trControl = fitControl)
```

```
## Loading required package: MASS
```

```r
stopCluster(cluster)
```

Estimate out of sample error from cross-validation. Accuracy is 51.7%, error is 48.3%. This confirms that random forest is a good choice for this analysis.

```r
confusionMatrix(modFit2)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 19.6  5.7  8.5  2.9  2.9
##          B  1.3  8.2  1.5  1.0  3.6
##          C  2.0  3.0  5.4  1.3  1.5
##          D  5.1  1.7  1.8 10.1  2.2
##          E  0.5  0.7  0.2  1.2  8.2
```

```r
accuracy <- 19.6 + 8.2 + 5.4 + 10.3 + 8.2
error2 = 100 - accuracy
error2
```

```
## [1] 48.3
```
