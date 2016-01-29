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

We find that a random forest model with 10-fold cross-validation generates a lower out of sample error than a linear discriminant analysis with 10-fold cross-validation trained on the same dataset.

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

Limit variables to those relating to accelaration on belt, forearm, arm, and dumbbell. Also include name of individual who performed the exercise, and the outcome variable.

```r
accell <- grep("accel", colnames(training))
training_accel <- training[,c(2, accell, which(colnames(training) == "classe"))]
```

Determine whethere there are any variables with missing values that can be removed from the dataset. Variance of accelleration in all areas missing the majority of observations; remove those variables.

```r
summary(training_accel)
```

```
##   user_name         total_accel_belt var_total_accel_belt
##  Length:19622       Min.   : 0.00    Min.   : 0.000      
##  Class :character   1st Qu.: 3.00    1st Qu.: 0.100      
##  Mode  :character   Median :17.00    Median : 0.200      
##                     Mean   :11.31    Mean   : 0.926      
##                     3rd Qu.:18.00    3rd Qu.: 0.300      
##                     Max.   :29.00    Max.   :16.500      
##                                      NA's   :19216       
##   accel_belt_x       accel_belt_y     accel_belt_z     total_accel_arm
##  Min.   :-120.000   Min.   :-69.00   Min.   :-275.00   Min.   : 1.00  
##  1st Qu.: -21.000   1st Qu.:  3.00   1st Qu.:-162.00   1st Qu.:17.00  
##  Median : -15.000   Median : 35.00   Median :-152.00   Median :27.00  
##  Mean   :  -5.595   Mean   : 30.15   Mean   : -72.59   Mean   :25.51  
##  3rd Qu.:  -5.000   3rd Qu.: 61.00   3rd Qu.:  27.00   3rd Qu.:33.00  
##  Max.   :  85.000   Max.   :164.00   Max.   : 105.00   Max.   :66.00  
##                                                                       
##  var_accel_arm     accel_arm_x       accel_arm_y      accel_arm_z     
##  Min.   :  0.00   Min.   :-404.00   Min.   :-318.0   Min.   :-636.00  
##  1st Qu.:  9.03   1st Qu.:-242.00   1st Qu.: -54.0   1st Qu.:-143.00  
##  Median : 40.61   Median : -44.00   Median :  14.0   Median : -47.00  
##  Mean   : 53.23   Mean   : -60.24   Mean   :  32.6   Mean   : -71.25  
##  3rd Qu.: 75.62   3rd Qu.:  84.00   3rd Qu.: 139.0   3rd Qu.:  23.00  
##  Max.   :331.70   Max.   : 437.00   Max.   : 308.0   Max.   : 292.00  
##  NA's   :19216                                                        
##  total_accel_dumbbell var_accel_dumbbell accel_dumbbell_x 
##  Min.   : 0.00        Min.   :  0.000    Min.   :-419.00  
##  1st Qu.: 4.00        1st Qu.:  0.378    1st Qu.: -50.00  
##  Median :10.00        Median :  1.000    Median :  -8.00  
##  Mean   :13.72        Mean   :  4.388    Mean   : -28.62  
##  3rd Qu.:19.00        3rd Qu.:  3.434    3rd Qu.:  11.00  
##  Max.   :58.00        Max.   :230.428    Max.   : 235.00  
##                       NA's   :19216                       
##  accel_dumbbell_y  accel_dumbbell_z  total_accel_forearm var_accel_forearm
##  Min.   :-189.00   Min.   :-334.00   Min.   :  0.00      Min.   :  0.000  
##  1st Qu.:  -8.00   1st Qu.:-142.00   1st Qu.: 29.00      1st Qu.:  6.759  
##  Median :  41.50   Median :  -1.00   Median : 36.00      Median : 21.165  
##  Mean   :  52.63   Mean   : -38.32   Mean   : 34.72      Mean   : 33.502  
##  3rd Qu.: 111.00   3rd Qu.:  38.00   3rd Qu.: 41.00      3rd Qu.: 51.240  
##  Max.   : 315.00   Max.   : 318.00   Max.   :108.00      Max.   :172.606  
##                                                          NA's   :19216    
##  accel_forearm_x   accel_forearm_y  accel_forearm_z      classe         
##  Min.   :-498.00   Min.   :-632.0   Min.   :-446.00   Length:19622      
##  1st Qu.:-178.00   1st Qu.:  57.0   1st Qu.:-182.00   Class :character  
##  Median : -57.00   Median : 201.0   Median : -39.00   Mode  :character  
##  Mean   : -61.65   Mean   : 163.7   Mean   : -55.29                     
##  3rd Qu.:  76.00   3rd Qu.: 312.0   3rd Qu.:  26.00                     
##  Max.   : 477.00   Max.   : 923.0   Max.   : 291.00                     
## 
```

```r
varAccell <- grep("^var", colnames(training_accel))
training_accel <- training_accel[,-varAccell]
```

Convert the user_name and outcome variables to factors to prepare for model building.

```r
training_accel$user_name = as.factor(training_accel$user_name)
training_accel$classe = as.factor(training_accel$classe)
```

Consider whether any variables are correlated, and should perhaps be subject to a principle components analysis.

```r
M <- abs(cor(training_accel[,-c(1, 18)]))
diag(M) <- 0
which(M > 0.8, arr.ind = T)
```

```
##                  row col
## accel_belt_y       3   1
## accel_belt_z       4   1
## total_accel_belt   1   3
## accel_belt_z       4   3
## total_accel_belt   1   4
## accel_belt_y       3   4
```

All but one variable relating to 'belt' measurements are highly correlated. Group all 'belt' variables into principle components. 

```r
training_accel_belt <- training_accel[, grep("belt", colnames(training_accel))]
preProc <- preProcess(training_accel_belt, method = "pca", thresh = 0.9)

# apply principal components to training set
train_beltPC <- predict(preProc, training_accel_belt)
```

Scale variables not entered into principle components.

```r
scaleObj <- preProcess(training_accel[, -c(1, 18)], method = c("center", "scale"))
training_scaled <- predict(scaleObj, training_accel[, -c(1, 18)])
```

Combine scaled variables, principle components, subject variable, and outcome variable into final training set to be used for the model.

```r
train_noBelt <- training_scaled[, -grep("belt", colnames(training_scaled))]
train_noBelt <- cbind(train_noBelt, training_accel[, c(1, 18)])
trainPC <- cbind(train_noBelt, train_beltPC)
```

## Predicitve model
Use random forest algorithm to model excercise data, due to its high accuracy. Due to high computational load, allow for parallel processing to speed the process.

Specify that 10-fold cross-validation should be used, include control parameters in train function.

```r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = T)

modFit <- train(classe ~., trainPC, method = "rf", trControl = fitControl)

stopCluster(cluster)
```

Based on the 10-fold cross-validation, evaluate the out of sample error. Accuracy across the folds is 94.2%, with 5.8% estimated out of sample error.

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
##          A 27.3  0.7  0.4  0.4  0.1
##          B  0.2 17.8  0.4  0.1  0.4
##          C  0.4  0.4 16.4  0.7  0.2
##          D  0.5  0.2  0.1 15.1  0.3
##          E  0.1  0.2  0.1  0.2 17.4
```

```r
accuracy <- 27.4 + 17.8 + 16.5 + 15.1 + 17.4
error <- 100 - accuracy
error
```

```
## [1] 5.8
```

## Prediction
Prepare the test set, applying same transformations and principle components anaysis as on the training set.

```r
accell_test <- grep("accel", colnames(testing))
testing_accel <- testing[,c(2, accell_test, 160)]
varAccell_test <- grep("^var", colnames(testing_accel))
testing_accel <- testing_accel[,-varAccell_test]

# PCA
testing_accel_belt <- testing_accel[, grep("belt", colnames(testing_accel))]
test_beltPC <- predict(preProc, testing_accel_belt)

# Scaling
scaleObj_test <- preProcess(testing_accel[, -c(1, 18)], method = c("center", "scale"))
testing_scaled <- predict(scaleObj_test, testing_accel[, -c(1, 18)])

# Combine
test_noBelt <- testing_scaled[, -grep("belt", colnames(testing_scaled))]
test_noBelt <- cbind(test_noBelt, testing_accel[, c(1, 18)])
testPC <- cbind(test_noBelt, test_beltPC)
testPC$user_name = as.factor(testPC$user_name)
```

Use random forest model to predict movement classification in test set.

```r
pred <- predict(modFit, testPC[, -14])
testPC$classe <- pred
testPC[, c(14, 17)]
```

```
##    problem_id classe
## 1           1      D
## 2           2      B
## 3           3      C
## 4           4      A
## 5           5      C
## 6           6      E
## 7           7      D
## 8           8      D
## 9           9      A
## 10         10      D
## 11         11      C
## 12         12      B
## 13         13      B
## 14         14      E
## 15         15      E
## 16         16      E
## 17         17      E
## 18         18      E
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


modFit2 <- train(classe ~., trainPC, method = "lda", trControl = fitControl)
```

```
## Loading required package: MASS
```

```r
stopCluster(cluster)
```

Estimate out of sample error from cross-validation. Accuracy is 52.7%, error is 47.3%. This confirms that random forest is a good choice for this analysis.

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
##          A 19.7  5.4  6.8  3.3  2.7
##          B  2.0  9.1  1.0  1.1  3.9
##          C  2.0  2.7  7.0  1.2  2.4
##          D  4.1  1.0  2.1  9.6  2.3
##          E  0.5  1.2  0.6  1.2  7.1
```

```r
accuracy <- 19.9 + 9 + 7 + 9.6 + 7.2
error2 = 100 - accuracy
error2
```

```
## [1] 47.3
```
