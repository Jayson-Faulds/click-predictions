library(data.table)
library(dplyr)
library(MLmetrics)
library(glmnet)
library(yardstick)

# load data sets
df <- fread("cleanedTraining.csv")
valdata <- fread("ProjectValidationData.csv")
df<- df[,-1]

# This is the same process of cleaning the validation data as in the other R scripts
names <- c('id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 
           'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16',
           'C17', 'C18', 'C19', 'C20', 'C21') 
colnames(valdata) <- names
valdata <- valdata[, -1] 

# bunch of left joins to get all of the reclassified columns into the validation data
newValData <- left_join(valdata, distinct_at(df[, c(5, 25)], .vars = 'site_id', .keep_all = T), by = 'site_id') 
newValData <- left_join(newValData, distinct_at(df[, c(6, 27)], .vars = 'site_domain', .keep_all = T), by = 'site_domain') 
newValData <- left_join(newValData, distinct_at(df[, c(7, 28)], .vars = 'site_category', .keep_all = T), by = 'site_category') 
newValData <- left_join(newValData, distinct_at(df[, c(8, 29)], .vars = 'app_id', .keep_all = T), by = 'app_id') 
newValData <- left_join(newValData, distinct_at(df[, c(9, 30)], .vars = 'app_domain', .keep_all = T), by = 'app_domain') 
newValData <- left_join(newValData, distinct_at(df[, c(10, 31)], .vars = 'app_category', .keep_all = T), by = 'app_category') 
newValData <- left_join(newValData, distinct_at(df[, c(11, 32)], .vars = 'device_id', .keep_all = T), by = 'device_id') 
newValData <- left_join(newValData, distinct_at(df[, c(12, 33)], .vars = 'device_ip', .keep_all = T), by = 'device_ip') 
newValData <- left_join(newValData, distinct_at(df[, c(13, 34)], .vars = 'device_model', .keep_all = T), by = 'device_model') 
newValData <- left_join(newValData, distinct_at(df[, c(16, 35)], .vars = 'C14', .keep_all = T), by = 'C14') 
newValData <- left_join(newValData, distinct_at(df[, c(19, 36)], .vars = 'C17', .keep_all = T), by = 'C17') 
newValData <- left_join(newValData, distinct_at(df[, c(21, 37)], .vars = 'C19', .keep_all = T), by = 'C19') 
newValData <- left_join(newValData, distinct_at(df[, c(22, 38)], .vars = 'C20', .keep_all = T), by = 'C20')
newValData <- left_join(newValData, distinct_at(df[, c(23, 39)], .vars = 'C21', .keep_all = T), by = 'C21')

# gets rid of the old columns with all of the factor levels
dflogTrain <- df[,c(1:4,14:15,17:18,20,25,27:39)]
dflogVal <- newValData[,c(1:4,14:15,17:18,20,24:37)]

# cleans the hour column in validation data
dflogVal$hour <- as.character(dflogVal$hour) 
dflogVal$hour <- substr(dflogVal$hour, 7, 8) 
library(stringr)

dflogTrain$hour <- str_pad(dflogTrain$hour, 2, pad = '0') # for some reason the training data does not have leading zeros anymore

# Now we factor every column in both datasets
dflogTrain <- sapply(dflogTrain,FUN=function(x) { as.factor(x)})
dflogVal <- sapply(dflogVal,FUN=function(x) { as.factor(x)})
cleanedTraining <- as.data.frame(dflogTrain)
cleanedValidation <- as.data.frame(dflogVal)

# replace all of the NAs created by left_join with our "low_freq" designation (probably should have written a function)
cleanedValidation$new_device_id[is.na(cleanedValidation$new_device_id)] <- 'low_freq'
cleanedValidation$new_site_id[is.na(cleanedValidation$new_site_id)] <- 'low_freq'
cleanedValidation$new_site_domain[is.na(cleanedValidation$new_site_domain)] <- 'low_freq'
cleanedValidation$new_site_category[is.na(cleanedValidation$new_site_category)] <- 'low_freq'
cleanedValidation$new_app_id[is.na(cleanedValidation$new_app_id)] <- 'low_freq'
cleanedValidation$new_app_domain[is.na(cleanedValidation$new_app_domain)] <- 'low_freq'
cleanedValidation$new_app_category[is.na(cleanedValidation$new_app_category)] <- 'low_freq'
cleanedValidation$new_device_ip[is.na(cleanedValidation$new_device_ip)] <- 'low_freq'
cleanedValidation$new_device_model[is.na(cleanedValidation$new_device_model)] <- 'low_freq'
cleanedValidation$new_C14[is.na(cleanedValidation$new_C14)] <- 'low_freq'
cleanedValidation$new_C17[is.na(cleanedValidation$new_C17)] <- 'low_freq'
cleanedValidation$new_C19[is.na(cleanedValidation$new_C19)] <- 'low_freq'
cleanedValidation$new_C20[is.na(cleanedValidation$new_C20)] <- 'low_freq'
cleanedValidation$new_C21[is.na(cleanedValidation$new_C21)] <- 'low_freq'

# fixes the levels of one of the columns to allow our model to run
levels(cleanedValidation$device_type) <- levels(cleanedTraining$device_type)

# Creates the dummies and response vectors
XTrain <- model.matrix(click ~ .,cleanedTraining)
XVal <- model.matrix(click ~ . ,cleanedValidation)
YTrain <- cleanedTraining$click
YVal <- cleanedValidation$click

# runs the lasso on a range of values for lambda
grid <- 10^seq(7,-2,length=200)
outLasso <- glmnet(XTrain,YTrain,alpha=1,lambda=grid,thresh=1e-12,family = 'binomial') # setting family=binomial does the logistic regression
YHat <- predict(outLasso,newx=XVal)

# Apply the logloss function from MLmetrics package (We were having trouble manually calculating logloss for some reason)
# We find the index of the best performing lambda and store that lambda value
loglossL <- apply(YHat, 2, FUN = LogLoss ,(as.integer(YVal)-1))
which.min(loglossL)
optimal_lambda <-outLasso$lambda[200]

# rerun the lasso on the optimal value for lambda (0.01), recalculate Log loss
out <- glmnet(XTrain,YTrain,alpha = 1,lambda = optimal_lambda,thresh = 1e-12,family = "binomial")
YhatLasso <- predict(out, newx = XVal,type = "response")
LogLoss(YhatLasso, as.integer(YVal)-1)

# logloss: 0.413

# Calculate precision and recall for our predictions
YhatLasso <- predict(out, newx = XVal, type = 'class')
YhatLasso <- as.vector(YhatLasso)
YhatLasso <- factor(YhatLasso)
prediction_df <- cbind(YVal, YhatLasso)
prediction_df <- as.data.frame(prediction_df)
prediction_df$YVal <- factor(prediction_df$YVal)
prediction_df$YhatLasso <- factor(prediction_df$YhatLasso)
precision(data = prediction_df, YVal, YhatLasso, estimator = 'binary')

# precision: 0.834 (precise when it does predict clicks)

# Calculate recall manually 
x <- prediction_df[prediction_df$YVal==2, ]
nrow(x[x$YhatLasso==2, ])/nrow(x)

# recall: 0.03 (really bad at recognizing clicks)


# Now we try to account for class imbalance
# We want to set the class weight in the glmnet call, in order to give more weight to clicks than non-clicks

# First we need a vector for weight, which contains the weight that we give to each individual observation
# We will give all clicks the same weight, and all non-clicks the same weight, with clicks having larger weights assigned
click_index <- as.numeric(cleanedTraining$click)
head(click_index)

# Set the weight for clicks equal to the ratio of non-clicks to clicks
obs_weights <- ifelse(click_index == 2, 4.874, 1)
head(obs_weights)

# rerun the lasso regression using our weights
out2 <- glmnet(XTrain,YTrain,alpha = 1,lambda = 0.01,thresh = 1e-12,family = "binomial", weights = obs_weights)
YhatLasso <- predict(out2, newx = XVal,type = "response")
LogLoss(YhatLasso, as.integer(YVal)-1)

# logloss: 0.612

# Here we get our class predictions instead of probabilities, and use the predictions to generate precision and recall stats
YhatLasso2 <- predict(out2, newx = XVal, type = 'class')
YhatLasso2 <- as.vector(YhatLasso2)
YhatLasso2 <- factor(YhatLasso2)
prediction_df <- cbind(YVal, YhatLasso2)
prediction_df <- as.data.frame(prediction_df)
prediction_df$YVal <- factor(prediction_df$YVal)
prediction_df$YhatLasso2 <- factor(prediction_df$YhatLasso2)
precision(data = prediction_df, YVal, YhatLasso2, estimator = 'binary')

# precision: 0.917

recall(data = prediction_df, YVal, YhatLasso2, estimator = 'binary')

# Recall: 0.578






