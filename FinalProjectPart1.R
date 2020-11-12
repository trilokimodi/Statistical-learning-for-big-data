setwd("D:/Masters Program Chalmers/Projects and Labs/SLBD/Final Project/Part 1")


# Packages and cbpalette --------------------------------------------------

install.packages("gglasso")
install.packages("e1071")
install.packages("pracma") # linspace
install.packages("clusterGeneration")
install.packages("glmnet")
install.packages("glm.predict")
install.packages("ROCR")
install.packages("mltools")
library(glm.predict)
library(e1071) # Skewness
library(caret)
library(pracma)   #Linspace
library(reshape2)
library(tidyverse)
library(ggplot2)
library(glmnet)
library(ROCR)
library(MASS)
library(randomForest)
cbPalette <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7","#b20000")


#Rename as per my convenience
xTrain <- X_train
xTest <- X_valid
yTrain <- y_train
yTest <- y_valid
rm(X_train,X_valid,y_valid,y_train)

#See if the labels are skewed - Yes
nZeros <- length(which(yTrain == 0))
nOnes <- length(which(yTrain == 1))

#As factor -1,1
yTrainlogit <- yTrain
yTrainlogit[which(yTrainlogit == 0)] <- -1
yTestClass <- yTest
yTestClass[which(yTestClass == 0)] <- -1
yTestClassfactor <- as.factor(yTestClass)

weights <- rep(1,times = nrow(yTrain))
weights[which(yTrain == 1)] <- nZeros/nOnes

yTrainFactor <- as.factor(yTrainlogit)
yTestFactor <- as.factor(yTest)

# Rename to DesignMatrices as matrix------------------------------------------------------

nFeatures <- ncol(xTrain)
nObservationTrain <- nrow(xTrain)
nObservationTest <- nrow(xTest)
designMatrixTrain <- matrix(1,nrow = nObservationTrain, ncol = nFeatures)
for (i in 1:nFeatures) {
  designMatrixTrain[,i] <- xTrain[,i]
}

designMatrixTest <- matrix(1,nrow = nObservationTest, ncol = nFeatures)
for (i in 1:nFeatures) {
  designMatrixTest[,i] <- xTest[,i]
}

designMatrixTrain <- scale(designMatrixTrain)
designMatrixTest <- scale(designMatrixTest)


# EDA ---------------------------------------------------------------------

kurtosisTrain <- apply(designMatrixTrain, 2, kurtosis)
h <- hist(kurtosisTrain,main = "Histogram of kurtosis training data",xlab = "Kurtosis value",col = "gray")
text(h$mids,h$counts,labels=h$counts, adj=c(0.5, -0.2))

skewnessTrain <- apply(designMatrixTrain, 2, e1071::skewness)
s <- hist(skewnessTrain,main = "Histogram of skewness training data",xlab = "Skewness value",col = "gray")
text(s$mids,s$counts,labels=s$counts, adj=c(0.5, -0.2))

designMatrixTrainTransform <- designMatrixTrain + 50 #A constant more than minimum of designmatrix
designMatrixTestTransform <- designMatrixTest + 50 #test and train for log transform

#Same columns are transformed, look at skewnessTrain
designMatrixTrainTransform[,which(abs(skewnessTrain) > 0.25)] <- 
  apply(designMatrixTrainTransform[,which(abs(skewnessTrain) > 0.25)],2,log)
designMatrixTestTransform[,which(abs(skewnessTrain) > 0.25)] <- 
  apply(designMatrixTestTransform[,which(abs(skewnessTrain) > 0.25)],2,log)

designMatrixTrainTransformBackup <- designMatrixTrainTransform
# Correlation in the designmatrix

correlatedMatrixcoeff <- cor(designMatrixTrain)
correlationDataFrame <- data.frame("V1" = rep(0,times = length(correlatedMatrixcoeff))
                                   , "V2" = rep(0,times = length(correlatedMatrixcoeff))
                                   ,"correlation" = rep(0,times = length(correlatedMatrixcoeff)))

correlationDataFrame$correlation = as.vector(t(correlatedMatrixcoeff))
correlationDataFrame$V1 = rep(c(1:nFeatures), each = nFeatures)
correlationDataFrame$V2 = rep(c(1:nFeatures),times = nFeatures)

# Sort by absolute correlation coeff and deleting 1's and repetitions
correlationDataFrame <- correlationDataFrame[
  order(abs(correlationDataFrame$correlation),decreasing = T),]
correlationDataFrame <- correlationDataFrame[-(1:nFeatures),]
toDelete <- seq(2, nrow(correlationDataFrame), 2)
correlationDataFrame <- correlationDataFrame[-toDelete,]



# Grouping based on collinearity ------------------------------------------

#GroupByCollinearity is the final true group

groupSize <- 80  #Just to order based on number of highly correlated features for a given feature
nGroups <- nFeatures/groupSize
groupByCollinearity <- rep(0,times = nFeatures)
occurrenceCollinearity <- rep(0, times = nFeatures)
k <- 1
temp <- 0
for (i in 1:nrow(correlationDataFrame)) {
  occurrenceCollinearity[correlationDataFrame$V1[i]] <- 
    occurrenceCollinearity[correlationDataFrame$V1[i]] + 1
  occurrenceCollinearity[correlationDataFrame$V2[i]] <- 
    occurrenceCollinearity[correlationDataFrame$V2[i]] + 1
  for (j in 1:nFeatures) {
    if (occurrenceCollinearity[j] == groupSize & groupByCollinearity[j] == 0) {
      groupByCollinearity[j] = k
      temp = temp + 1
      if (temp == groupSize) {
        temp = 0
        k = k + 1
      }
    }
  }
}
groupByCollinearity
unique(groupByCollinearity)

# Reshuffling of design matrix based on true group ------------------------

designMatrixTrainTemp <- designMatrixTrain
designMatrixTestTemp <- designMatrixTest
designMatrixTrainTransformTemp <- designMatrixTrainTransform
designMatrixTestTransformTemp <- designMatrixTestTransform
tempVector <- rep(0, times = groupSize)
for (i in 1:nGroups) {
  tempVector <- which(groupByCollinearity == i)
  designMatrixTrainTemp[,c(((i-1)*groupSize + 1):(i*groupSize))] <- 
    designMatrixTrain[,tempVector]
  designMatrixTestTemp[,c(((i-1)*groupSize + 1):(i*groupSize))] <- 
    designMatrixTest[,tempVector]
  designMatrixTrainTransformTemp[,c(((i-1)*groupSize + 1):(i*groupSize))] <- 
    designMatrixTrainTransform[,tempVector]
  designMatrixTestTransformTemp[,c(((i-1)*groupSize + 1):(i*groupSize))] <- 
    designMatrixTestTransform[,tempVector]
}

designMatrixTrain <- designMatrixTrainTemp   #This is the final design matrix for true model
designMatrixTest <- designMatrixTestTemp
designMatrixTrainTransform <- designMatrixTrainTransformTemp
designMatrixTestTransform <- designMatrixTestTransformTemp

correlatedMatrixcoeff <- cor(designMatrixTrain)


# Multiple runs -----------------------------------------------------------

# Sclaed data
{
  designMatrixToUseTrain <- designMatrixTrain
  designMatrixToUseTest <- designMatrixTest
  flag = 1
}

# Scaled and transformed data Group
{
  designMatrixToUseTrain <- designMatrixTrainTransform
  designMatrixToUseTest <- designMatrixTestTransform
  flag = 2
}

alphaVector <- seq(from = 0.1,to = 1,by = 0.1)
nMaxRuns <- 50


confMatrixClass <- vector(mode = "list",length = length(alphaVector))
confMatrixResponse <- vector(mode = "list",length = length(alphaVector))
thresholdProbability <- vector(mode = "list",length = length(alphaVector))
influentialColumns <- vector(mode = "list",length = length(alphaVector))

for (iAlpha in 1:length(alphaVector)) {
  confMatrixClass[[iAlpha]] <- vector(mode = "list",length = nMaxRuns)
  thresholdProbability[[iAlpha]] <- rep(0,times = nMaxRuns)
  confMatrixResponse[[iAlpha]] <- vector(mode = "list",length = nMaxRuns)
  influentialColumns[[iAlpha]] <- rep(0,times = nFeatures)
}

for (iAlpha in 1:length(alphaVector)) {
  iRun <- 1
  while (iRun <= nMaxRuns) {
    
    #alpha = 0 implies ridge regression
    #alpha = 1 implies lasso
    #alpha = 0 to 1 implies elnet and it is useful for correlated data
      
    glmfit <- cv.glmnet(
      x = designMatrixToUseTrain,y = yTrainFactor,
      alpha = alphaVector[iAlpha],
      family = "binomial",
      type.measure = "deviance",
      weights = weights,
      nfolds = 17)
    
    yPredClass <- predict(glmfit,newx = designMatrixToUseTest,s = c("lambda.min"), type = "class")
    confMatrixClass[[iAlpha]][[iRun]] <- 
      confusionMatrix(data = as.factor(yPredClass),reference = yTestClassfactor,positive = "1")
    
    #For type response we want to maximize both sensitivity and specificity
    #We know that FPR is 1 - specificity
    #Therefore, the minimum value of squared sum of TPR and 1 - FPR will give 
    #the optimal threshold for which both TPR i.e. sensitivity
    #and (1 - FPR) i.e. specificity will be maximized
    #Why min because these values are less than 1. 
    
    yPredResponse <- predict(glmfit,newx = designMatrixToUseTest,s = c("lambda.min"), type = "response")
    ROCRpred <- prediction(yPredResponse, yTestFactor)
    ROCRperf <- performance(ROCRpred, "tpr", "fpr")
    #plot(ROCRperf,main = "ROC Curve, L1 Lasso, Run = 50",colorize = T)
    
    SumSensSpec <- (1 - ROCRperf@x.values[[1]]) + (ROCRperf@y.values[[1]])
    maxSumSensSpec <- max(SumSensSpec)
    maxSumSensSpecIndex <- which(maxSumSensSpec == SumSensSpec)
    thresholdProbability[[iAlpha]][iRun] <- ROCRperf@alpha.values[[1]][maxSumSensSpecIndex]
    
    yPredBinomial <- yPredResponse
    yPredBinomial[which(yPredResponse < thresholdProbability[[iAlpha]][iRun])] = 0
    yPredBinomial[which(yPredResponse >= thresholdProbability[[iAlpha]][iRun])] = 1
    yPredBinomial <- as.factor(yPredBinomial)
    confMatrixResponse[[iAlpha]][[iRun]] <- 
      confusionMatrix(data = yPredBinomial,reference = yTestFactor,positive = "1")
    
    coeff <- coef.glmnet(glmfit,exact = F, s = c("lambda.min"))
    coeff <- coeff[-1]
    coeffTrueAsFactorList <- coeff
    coeffTrueAsFactorList[which(coeff != 0)] <- 1
    
    coeffTrueAsFactorList <- factor(coeffTrueAsFactorList,levels = c(0,1))
    influentialColumns[[iAlpha]][which(coeffTrueAsFactorList == 1)] = 
      influentialColumns[[iAlpha]][which(coeffTrueAsFactorList == 1)] + 1
    
    iRun <- iRun + 1
  }
}

meanThresholdProb <- rep(0,length(alphaVector))
for (iAlpha in 1:length(alphaVector)) {
  meanThresholdProb[iAlpha] <- apply(as.matrix(thresholdProbability[[iAlpha]]), 2, mean)
}
mean(meanThresholdProb)  #For report


# Plot the measures -------------------------------------------------------

#3 for high, medium, low influential columns
nInfluentialColumns <- matrix(0,nrow = 4,ncol = length(alphaVector))  

accuracyResponse <- rep(0,times = length(alphaVector))
f1ScoreResponse <- rep(0,times = length(alphaVector))
sensitivityResponse <- rep(0,times = length(alphaVector))
precisionResponse <- rep(0,times = length(alphaVector))
mccResponse <- rep(0,times = length(alphaVector))

influentialColumnsOverall <- rep(0,times = nFeatures)

for (iAlpha in 1:length(alphaVector)) {
  for (iRun in 1:nMaxRuns) {
    accuracyResponse[iAlpha] = accuracyResponse[iAlpha] + 
      confMatrixResponse[[iAlpha]][[iRun]][["overall"]][["Accuracy"]]
    f1ScoreResponse[iAlpha] = f1ScoreResponse[iAlpha] + 
      confMatrixResponse[[iAlpha]][[iRun]][["byClass"]][["F1"]]
    sensitivityResponse[iAlpha] = sensitivityResponse[iAlpha] + 
      confMatrixResponse[[iAlpha]][[iRun]][["byClass"]][["Sensitivity"]]
    precisionResponse[iAlpha] = precisionResponse[iAlpha] + 
      confMatrixResponse[[iAlpha]][[iRun]][["byClass"]][["Precision"]]
    mccResponse[iAlpha] = mccResponse[iAlpha] + ((confMatrixResponse[[iAlpha]][[iRun]][["table"]][4] * 
              confMatrixResponse[[iAlpha]][[iRun]][["table"]][1] - 
              confMatrixResponse[[iAlpha]][[iRun]][["table"]][3] * 
              confMatrixResponse[[iAlpha]][[iRun]][["table"]][2]))/
      sqrt((confMatrixResponse[[iAlpha]][[iRun]][["table"]][2] + 
              confMatrixResponse[[iAlpha]][[iRun]][["table"]][4]) *
             (confMatrixResponse[[iAlpha]][[iRun]][["table"]][3] + 
                confMatrixResponse[[iAlpha]][[iRun]][["table"]][4]) * 
             (confMatrixResponse[[iAlpha]][[iRun]][["table"]][2] +
                confMatrixResponse[[iAlpha]][[iRun]][["table"]][1]) *
             (confMatrixResponse[[iAlpha]][[iRun]][["table"]][3] +
                confMatrixResponse[[iAlpha]][[iRun]][["table"]][1]))
    
  }
  accuracyResponse[iAlpha] <- accuracyResponse[iAlpha]/nMaxRuns
  f1ScoreResponse[iAlpha] <- f1ScoreResponse[iAlpha]/nMaxRuns
  sensitivityResponse[iAlpha] <- sensitivityResponse[iAlpha]/nMaxRuns
  precisionResponse[iAlpha] <- precisionResponse[iAlpha]/nMaxRuns
  mccResponse[iAlpha] <- mccResponse[iAlpha]/nMaxRuns
  
  nInfluentialColumns[1,iAlpha] <- length(which(influentialColumns[[iAlpha]] == 50))
  nInfluentialColumns[2,iAlpha] <- length(which(influentialColumns[[iAlpha]] >= 45 & influentialColumns[[iAlpha]] < 50))
  nInfluentialColumns[3,iAlpha] <- length(which(influentialColumns[[iAlpha]] >= 25 & influentialColumns[[iAlpha]] < 45))
  nInfluentialColumns[4,iAlpha] <- length(which(influentialColumns[[iAlpha]] >= 10 & influentialColumns[[iAlpha]] < 25))#Stacked bar chart
  
  influentialColumnsOverall <- influentialColumnsOverall + influentialColumns[[iAlpha]]
}
influentialColumnsOverall <- influentialColumnsOverall/nMaxRuns

accuracyClass <- rep(0,times = length(alphaVector))
f1ScoreClass <- rep(0,times = length(alphaVector))
sensitivityClass <- rep(0,times = length(alphaVector))
precisionClass <- rep(0,times = length(alphaVector))
mccClass <- rep(0,times = length(alphaVector))

for (iAlpha in 1:length(alphaVector)) {
    for (iRun in 1:nMaxRuns) {
      accuracyClass[iAlpha] = accuracyClass[iAlpha] + 
        confMatrixClass[[iAlpha]][[iRun]][["overall"]][["Accuracy"]]
      f1ScoreClass[iAlpha] = f1ScoreClass[iAlpha] + 
        confMatrixClass[[iAlpha]][[iRun]][["byClass"]][["F1"]]
      sensitivityClass[iAlpha] = sensitivityClass[iAlpha] + 
        confMatrixClass[[iAlpha]][[iRun]][["byClass"]][["Sensitivity"]]
      precisionClass[iAlpha] = precisionClass[iAlpha] + 
        confMatrixClass[[iAlpha]][[iRun]][["byClass"]][["Precision"]]
      mccClass[iAlpha] = ((confMatrixClass[[iAlpha]][[iRun]][["table"]][4] * 
                                confMatrixClass[[iAlpha]][[iRun]][["table"]][1] - 
                                confMatrixClass[[iAlpha]][[iRun]][["table"]][3] * 
                                confMatrixClass[[iAlpha]][[iRun]][["table"]][2]))/
        sqrt((confMatrixClass[[iAlpha]][[iRun]][["table"]][2] + 
                confMatrixClass[[iAlpha]][[iRun]][["table"]][4]) *
               (confMatrixClass[[iAlpha]][[iRun]][["table"]][3] + 
                  confMatrixClass[[iAlpha]][[iRun]][["table"]][4]) * 
               (confMatrixClass[[iAlpha]][[iRun]][["table"]][2] +
                  confMatrixClass[[iAlpha]][[iRun]][["table"]][1]) *
               (confMatrixClass[[iAlpha]][[iRun]][["table"]][3] +
                  confMatrixClass[[iAlpha]][[iRun]][["table"]][1]))
    }
  accuracyClass[iAlpha] <- accuracyClass[iAlpha]/nMaxRuns
  f1ScoreClass[iAlpha] <- f1ScoreClass[iAlpha]/nMaxRuns
  sensitivityClass[iAlpha] <- sensitivityClass[iAlpha]/nMaxRuns
  precisionClass[iAlpha] <- precisionClass[iAlpha]/nMaxRuns
  mccClass[iAlpha] <- mccClass[iAlpha]/nMaxRuns
}


plotframe <- vector(mode = "list", length = 2)
plotframe[[1]] <- data.frame("Alpha" = alphaVector,
                                   "accuracyVector" = accuracyResponse,
                                   "precisionVector" = precisionResponse,
                                   "sensitivityVector" = sensitivityResponse,
                                   "f1Vector" = f1ScoreResponse,
                                    "mccVector" = mccResponse)

plotframe[[2]] <- data.frame("Alpha" = alphaVector,
                             "accuracyVector" = accuracyClass,
                             "precisionVector" = precisionClass,
                             "sensitivityVector" = sensitivityClass,
                             "f1Vector" = f1ScoreClass,
                             "mccVector" = mccClass)


for (iList in 1:length(plotframe)) {
  plotframe[[iList]] <- plotframe[[iList]][complete.cases(plotframe[[iList]]), ]
}

for (iList in 1:length(plotframe)) {
  ggplo <- ggplot(data = plotframe[[iList]],
                  aes(x = alphaVector)) +
    geom_point(aes(y = plotframe[[iList]]$accuracyVector, color = cbPalette[3])) +
    geom_point(aes(y = plotframe[[iList]]$precisionVector, color = cbPalette[4])) + 
    geom_point(aes(y = plotframe[[iList]]$sensitivityVector, color = cbPalette[5])) + 
    geom_line(y = plotframe[[iList]]$accuracyVector, colour = cbPalette[3], lwd = 1.5) +
    geom_line(y = plotframe[[iList]]$precisionVector, colour = cbPalette[4], lwd = 1.5) +
    geom_line(y = plotframe[[iList]]$sensitivityVector, colour = cbPalette[5], lwd = 1.5) +
    theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    scale_x_continuous(labels = alphaVector, breaks = alphaVector) + 
    labs(x = "Alpha",y = "Evaluation Metrics", title = "Evaluation Metrics for Lasso and Elnet", color = "Evaluation Metric") + 
    theme(legend.position = "top") + 
    scale_color_manual(labels = c("Accuracy","Precision","Recall"), values = cbPalette[3:5])
  print(ggplo)
}

for (iList in 1:length(plotframe)) {
  ggplo <- ggplot(data = plotframe[[iList]],
                  aes(x = alphaVector)) +
    geom_point(aes(y = plotframe[[iList]]$mccVector, color = cbPalette[3])) +
    geom_point(aes(y = plotframe[[iList]]$f1Vector, color = cbPalette[5])) + 
    geom_line(y = plotframe[[iList]]$mccVector, colour = cbPalette[3], lwd = 1.5) +
    geom_line(y = plotframe[[iList]]$f1Vector, colour = cbPalette[5], lwd = 1.5) + 
    theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    scale_x_continuous(labels = alphaVector, breaks = alphaVector) + 
    labs(x = "Alpha Value",y = "Evaluation Metrics", title = "Evaluation Metrics for Lasso and Elnet", color = "Evaluation Metric") + 
    theme(legend.position = "top") + 
    scale_color_manual(labels = c("MCC Score","F1 Score"), values = cbPalette[c(3,5)])
  print(ggplo)
}


#Stacked bar chart

barplot(height = nInfluentialColumns,names.arg = alphaVector,legend.text = c("Dominant = 50","50 > High > 45","45 > Medium > 25","25 > Low > 10"),
          beside = F,main = "Number of nonzero features by influence",
        xlab = "Alpha",ylab = "Number of features - cumulative",col = cbPalette[1:4])

group <- function(index) { #These group number are motivated by cbpallete. Typically
  #They represent feature which is usually highly to lowly correlated with others respectively
  if (index <= 80) {
    group = 9  # High red
  }
  else if (index > 80 & index < 240) {
    group = 5  #Medium Yellow
  }
  else
    group = 6 #Low Blue
}

influentialColumnsOverallPlot <- matrix(1,nrow = length(which(influentialColumnsOverall > 0)),ncol = 3)
influentialColumnsOverallPlot[,1] <- influentialColumnsOverall[which(influentialColumnsOverall > 0)]
influentialColumnsOverallPlot[,2] <- which(influentialColumnsOverall > 0)
influentialColumnsOverallPlot[,3] <- apply(as.matrix(influentialColumnsOverallPlot[,2]), 1, group)
influentialColumnsOverallPlot <- as.data.frame(influentialColumnsOverallPlot)
influentialColumnsOverallPlot <- influentialColumnsOverallPlot[
  order(abs(influentialColumnsOverallPlot$V1),decreasing = T),]
influentialColumnsOverallPlot <- as.matrix(influentialColumnsOverallPlot)

barplot(height = influentialColumnsOverallPlot[,1],
        names.arg = influentialColumnsOverallPlot[,2],las = 2,border = F,
        main = paste("Feature dominance and their likely correlation with other features\nTotal features which were non-zero in atleast 1 value of alpha = ",nrow(influentialColumnsOverallPlot),
                     "\nTotal features that were non-zero in atleast 8 values of alpha = ",length(which(influentialColumnsOverall >= 8))),
        xlab = "Feature number",ylab = "Average frequency of feature", col = cbPalette[influentialColumnsOverallPlot[,3]])

legend("topright", legend = c("High", "Mediocre","Negligible"), fill = cbPalette[c(9,5,6)])


# Most important features using wrapping method ----------------------------

finalFeatures <- which(influentialColumnsOverall >= 8)

trainTestData <- data.frame(rbind(
  designMatrixToUseTrain[,finalFeatures],designMatrixToUseTest[,finalFeatures]))
trainindices = c(1:nrow(designMatrixToUseTrain))
testindices = c(nrow(designMatrixToUseTrain) + 1 : nrow(designMatrixToUseTest))

finalFullModel <- glm(yTrain ~ .,family=binomial(link=logit),
                  data = trainTestData[trainindices,])
yPred <- predict(finalFullModel,newdata = trainTestData[-trainindices,],type = "response")
yPredFactor <- yPred
yPredFactor <- ifelse(yPredFactor > 0.5,"1","0")
yPredFactor <- as.factor(yPredFactor)

cf1 <- confusionMatrix(data = yPredFactor,reference = yTestFactor,positive = "1")
mcc1 <- ((cf1[["table"]][4] * cf1[["table"]][1] - 
                                                            cf1[["table"]][3] * 
                                                            cf1[["table"]][2]))/
  sqrt((cf1[["table"]][2] + 
          cf1[["table"]][4]) *
         (cf1[["table"]][3] + 
            cf1[["table"]][4]) * 
         (cf1[["table"]][2] +
            cf1[["table"]][1]) *
         (cf1[["table"]][3] +
            cf1[["table"]][1]))

modelWrap <- step(object = finalFullModel,direction = "backward",trace = T)
yPredWrap <- predict(object = modelWrap,newdata = trainTestData[-trainindices,],type = "response")
yPredWrapFactor <- yPredWrap
yPredWrapFactor <- ifelse(yPredWrapFactor > 0.5,"1","0")
yPredWrapFactor <- as.factor(yPredWrapFactor)

cf2 <- confusionMatrix(data = yPredWrapFactor,reference = yTestFactor,positive = "1")
mcc2 <- ((cf2[["table"]][4] * cf2[["table"]][1] - 
            cf2[["table"]][3] * 
            cf2[["table"]][2]))/
  sqrt((cf2[["table"]][2] + 
          cf2[["table"]][4]) *
         (cf2[["table"]][3] + 
            cf2[["table"]][4]) * 
         (cf2[["table"]][2] +
            cf2[["table"]][1]) *
         (cf2[["table"]][3] +
            cf2[["table"]][1]))

getFeatureNumberFunction <- function(input){sF <- as.numeric(substring(input,2))}
featureWrap <- apply(as.matrix(names(modelWrap[["model"]][,-1])), 1, getFeatureNumberFunction)

featureWrap <- finalFeatures[featureWrap]  #Feature number extraction in transformed data
featureWrap

# Retain true feature number ----------------------------------------------
getFeatureNumberFunction <- function(input){sF <- as.numeric(substring(input,2))}
featureWrap <- apply(as.matrix(names(modelWrap[["model"]][,-1])), 1, getFeatureNumberFunction)

featureWrap <- finalFeatures[featureWrap]  #Feature number extraction in transformed data
matchedFeatures <- rep(0,length(featureWrap))

for (iCheck in 1:length(featureWrap)) {
  for (jCheck in 1:nFeatures) {
    if (all(designMatrixTrainTransformBackup[,jCheck] == designMatrixToUseTrain[,featureWrap[iCheck]])) {
      matchedFeatures[iCheck] <- jCheck
    }
  }
}

sort(matchedFeatures)



