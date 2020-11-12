setwd("D:/Masters Program Chalmers/Projects and Labs/SLBD/Final Project/Part 3")

# Packages and cbpalette --------------------------------------------------

install.packages("e1071")
install.packages("bsts") #Geometric sequece
library(bsts)
library(e1071)
library(caret)
library(tidyverse)
library(ggplot2)
library(glmnet)

cbPalette <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")



# basic DesignMatrix ------------------------------------------------------

set.seed(7)
nFeatures <- 50
nObservationsVector <- GeometricSequence(7, initial.value = 100, discount.factor = 2)

targetSNR <- 1
groupSize <- 5
set.seed(7)

nMaxSets <- 50

tprAdaptiveSet <- vector(mode = "list",length = nMaxSets)
tnrAdaptiveSet <- vector(mode = "list",length = nMaxSets)
mccAdaptiveSet <- vector(mode = "list",length = nMaxSets)

tprLassoSet <- vector(mode = "list",length = nMaxSets)
tnrLassoSet <- vector(mode = "list",length = nMaxSets)
mccLassoSet <- vector(mode = "list",length = nMaxSets)


for (iset in 1:nMaxSets) {
  
betaAll <- c(rnorm(groupSize, mean=0, sd=1),rep(0,times = nFeatures - groupSize))

betaReference <- betaAll 
betaReference[which(betaAll != 0)] = 1
betaReference <- as.factor(betaReference)

trueModelList <- vector(mode = "list",length = length(nObservationsVector))
designMatrixList <- vector(mode = "list",length = length(nObservationsVector))

#Final Y
for (iList in 1:length(nObservationsVector)) {
  designMatrixList[[iList]] <- matrix(1,nrow = nObservationsVector[iList], ncol = nFeatures)
  for (i in 1:nFeatures) {
    designMatrixList[[iList]][,i] <- rnorm(n = nObservationsVector[iList],mean = 0,
                                           sd = 1/sqrt(nFeatures))
  }
  trueModelList[[iList]] = designMatrixList[[iList]] %*% betaAll
  noiseSigma <- sqrt(sum(trueModelList[[iList]]*trueModelList[[iList]])
                     /(nObservationsVector[iList]-1))/targetSNR
  noise <- rnorm(n = nObservationsVector[iList],mean = 0, sd = noiseSigma)
  trueModelList[[iList]] <- trueModelList[[iList]] + noise 
  #designMatrixList[[iList]] <- as.data.frame(designMatrixList[[iList]])
}


#Gamma and variable initialization
gammaAdaptive <- c(0.25,0.5,1,2,4,8)
nMaxRuns <- 50

weightsAdaptive <- vector(mode = "list", length = length(nObservationsVector))
lambdaLasso <- vector(mode = "list", length = length(nObservationsVector))
confMatrixLasso <- vector(mode = "list", length = length(nObservationsVector))
confMatrixAdaptive <- vector(mode = "list", length = length(nObservationsVector))
lambdaAdaptive <- vector(mode = "list", length = length(nObservationsVector))

for (iList in 1:length(nObservationsVector)) {
  weightsAdaptive[[iList]] <- vector(mode = "list",length = length(gammaAdaptive))
  lambdaLasso[[iList]] <- vector(mode = "list",length = nMaxRuns)
  confMatrixLasso[[iList]] <- vector(mode = "list",length = nMaxRuns)
  for (iGamma in 1:length(gammaAdaptive)) {
    weightsAdaptive[[iList]][[iGamma]] <- rep(0,times = nFeatures)
    lambdaAdaptive[[iList]][[iGamma]] <- vector(mode = "list",length = nMaxRuns)
    confMatrixAdaptive[[iList]][[iGamma]] <- vector(mode = "list",length = nMaxRuns)
  }
}

designMatrixTransformed <- vector(mode = "list", length = length(nObservationsVector))
for (iList in 1:length(nObservationsVector)) {
  designMatrixTransformed[[iList]] <- vector(mode = "list",length = length(gammaAdaptive))
}

# Method - 2 --------------------------------------------------------------

for (iList in 1:length(nObservationsVector)) {
  linearModel <- lm(trueModelList[[iList]] ~ .,data = as.data.frame(designMatrixList[[iList]]))
    iRun <- 1
    while (iRun <= nMaxRuns) {
      modelLasso <- cv.glmnet(x = designMatrixList[[iList]],y = trueModelList[[iList]],alpha = 1)
      lambdaLasso[[iList]][[iRun]] <- modelLasso[["lambda.min"]]
      coeffLasso <- coef.glmnet(modelLasso,s = c("lambda.1se"))
      coeffLasso <- coeffLasso[-1]
      
      coeffLassoNonZero <- coeffLasso
      coeffLassoNonZero[which(coeffLasso != 0)] <- 1
      coeffLassoNonZero <- as.factor(coeffLassoNonZero)
      confMatrixLasso[[iList]][[iRun]] <- 
        confusionMatrix(data = coeffLassoNonZero,reference = betaReference,positive = "1")
      
      for (iGamma in 1:length(gammaAdaptive)) {
        weightsAdaptive[[iList]][[iGamma]] <- 
          1/(unname(abs(linearModel[["coefficients"]][-1]))^gammaAdaptive[iGamma])
        designMatrixTransformed[[iList]][[iGamma]] <- apply(
          designMatrixList[[iList]],2,FUN = "/",weightsAdaptive[[iList]][[iGamma]])
        
        modelAdaptive <- cv.glmnet(x = designMatrixTransformed[[iList]][[iGamma]],y = trueModelList[[iList]],alpha = 1)
        lambdaAdaptive[[iList]][[iGamma]][[iRun]] <- modelAdaptive[["lambda.min"]]
        coeffAdaptive <- coef.glmnet(modelAdaptive,s = c("lambda.min"))
        coeffAdaptive <- coeffAdaptive[-1]
        coeffAdaptive <- (1/weightsAdaptive[[iList]][[iGamma]])*coeffAdaptive
        
        coeffAdaptiveNonZero <- coeffAdaptive
        coeffAdaptiveNonZero[which(coeffAdaptive != 0)] <- 1
        coeffAdaptiveNonZero <- as.factor(coeffAdaptiveNonZero)
        confMatrixAdaptive[[iList]][[iGamma]][[iRun]] <- 
          confusionMatrix(data = coeffAdaptiveNonZero,reference = betaReference,positive = "1")
      }
      iRun <- iRun + 1
  }
}


# For plots ---------------------------------------------------------------

tprAdaptive <- vector(mode = "list", length = length(nObservationsVector))
tnrAdaptive <- vector(mode = "list", length = length(nObservationsVector))
mccAdaptive <- vector(mode = "list", length = length(nObservationsVector))
for (iList in 1:length(nObservationsVector)) {
  tnrAdaptive[[iList]] <- rep(0,length(gammaAdaptive))
  tprAdaptive[[iList]] <- rep(0,length(gammaAdaptive))
  mccAdaptive[[iList]] <- rep(0,length(gammaAdaptive))
}

for (iList in 1:length(nObservationsVector)) {
  for (iGamma in 1:length(gammaAdaptive)) {
    for (iRun in 1:nMaxRuns) {
      tprAdaptive[[iList]][iGamma] <- tprAdaptive[[iList]][iGamma] + 
        confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["byClass"]][["Sensitivity"]]
      tnrAdaptive[[iList]][iGamma] <- tnrAdaptive[[iList]][iGamma] + 
        confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["byClass"]][["Specificity"]]
      mccAdaptive[[iList]][iGamma] <- mccAdaptive[[iList]][iGamma] + 
        ((confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][4] * 
                                                                         confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][1] - 
                                                                         confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][3] * 
                                                                         confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][2]))/
        sqrt((confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][2] + 
                confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][4]) *
               (confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][3] + 
                  confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][4]) * 
               (confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][2] +
                  confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][1]) *
               (confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][3] +
                  confMatrixAdaptive[[iList]][[iGamma]][[iRun]][["table"]][1]))
    }
    tprAdaptive[[iList]][iGamma] <- tprAdaptive[[iList]][iGamma]/nMaxRuns
    tnrAdaptive[[iList]][iGamma] <- tnrAdaptive[[iList]][iGamma]/nMaxRuns
    mccAdaptive[[iList]][iGamma] <- mccAdaptive[[iList]][iGamma]/nMaxRuns
  }
}

tnrLasso <- rep(0,length(nObservationsVector))
tprLasso <- rep(0,length(nObservationsVector))
mccLasso <- rep(0,length(nObservationsVector))

for (iList in 1:length(nObservationsVector)) {
    for (iRun in 1:nMaxRuns) {
      tprLasso[iList] <- tprLasso[iList] + 
        confMatrixLasso[[iList]][[iRun]][["byClass"]][["Sensitivity"]]
      tnrLasso[iList] <- tnrLasso[iList] + 
        confMatrixLasso[[iList]][[iRun]][["byClass"]][["Specificity"]]
      mccLasso[iList] <- mccLasso[iList] + ((confMatrixLasso[[iList]][[iRun]][["table"]][4] * 
                                          confMatrixLasso[[iList]][[iRun]][["table"]][1] - 
                                          confMatrixLasso[[iList]][[iRun]][["table"]][3] * 
                                          confMatrixLasso[[iList]][[iRun]][["table"]][2]))/
        sqrt((confMatrixLasso[[iList]][[iRun]][["table"]][2] + 
                confMatrixLasso[[iList]][[iRun]][["table"]][4]) *
               (confMatrixLasso[[iList]][[iRun]][["table"]][3] + 
                  confMatrixLasso[[iList]][[iRun]][["table"]][4]) * 
               (confMatrixLasso[[iList]][[iRun]][["table"]][2] +
                  confMatrixLasso[[iList]][[iRun]][["table"]][1]) *
               (confMatrixLasso[[iList]][[iRun]][["table"]][3] +
                  confMatrixLasso[[iList]][[iRun]][["table"]][1]))
    }
    tprLasso[iList] <- tprLasso[iList]/nMaxRuns
    tnrLasso[iList] <- tnrLasso[iList]/nMaxRuns
    mccLasso[iList] <- mccLasso[iList]/nMaxRuns
}

tprAdaptiveSet[[iset]] <- tprAdaptive
tnrAdaptiveSet[[iset]] <- tnrAdaptive
mccAdaptiveSet[[iset]] <- mccAdaptive

tprLassoSet[[iset]] <- tprLasso
tnrLassoSet[[iset]] <- tnrLasso
mccLassoSet[[iset]] <- mccLasso

}


#End of big for

tnrLasso <- rep(0,length(nObservationsVector))
tprLasso <- rep(0,length(nObservationsVector))
mccLasso <- rep(0,length(nObservationsVector))

for (iLasso in 1:length(tprLasso)) {
  for (iset in 1:nMaxSets) {
    tprLasso[iLasso] <- tprLasso[iLasso] + tprLassoSet[[iset]][iLasso]
    tnrLasso[iLasso] <- tnrLasso[iLasso] + tnrLassoSet[[iset]][iLasso]
    mccLasso[iLasso] <- mccLasso[iLasso] + mccLassoSet[[iset]][iLasso]
  }
  tprLasso[iLasso] <- tprLasso[iLasso]/nMaxSets
  tnrLasso[iLasso] <- tnrLasso[iLasso]/nMaxSets
  mccLasso[iLasso] <- mccLasso[iLasso]/nMaxSets
}



tprAdaptive <- vector(mode = "list", length = length(nObservationsVector))
tnrAdaptive <- vector(mode = "list", length = length(nObservationsVector))
mccAdaptive <- vector(mode = "list", length = length(nObservationsVector))
for (iList in 1:length(nObservationsVector)) {
  tnrAdaptive[[iList]] <- rep(0,length(gammaAdaptive))
  tprAdaptive[[iList]] <- rep(0,length(gammaAdaptive))
  mccAdaptive[[iList]] <- rep(0,length(gammaAdaptive))
}


for (iList in 1:length(nObservationsVector)) {
  for (iGamma in 1:length(gammaAdaptive)) {
    for (iSet in 1:nMaxSets) {
      tprAdaptive[[iList]][iGamma] <- tprAdaptive[[iList]][iGamma] + tprAdaptiveSet[[iSet]][[iList]][iGamma]
      tnrAdaptive[[iList]][iGamma] <- tnrAdaptive[[iList]][iGamma] + tnrAdaptiveSet[[iSet]][[iList]][iGamma]
      mccAdaptive[[iList]][iGamma] <- mccAdaptive[[iList]][iGamma] + mccAdaptiveSet[[iSet]][[iList]][iGamma]
    }
    tprAdaptive[[iList]][iGamma] <- tprAdaptive[[iList]][iGamma]/nMaxSets
    tnrAdaptive[[iList]][iGamma] <- tnrAdaptive[[iList]][iGamma]/nMaxSets
    mccAdaptive[[iList]][iGamma] <- mccAdaptive[[iList]][iGamma]/nMaxSets
  }
}


plotframe <- vector(mode = "list", length = (length(nObservationsVector) + 1))

for (iList in 1:length(nObservationsVector)) {
  plotframe[[iList]] <- data.frame("Gamma1" = gammaAdaptive,
                                   "TPR" = tprAdaptive[[iList]],
                                   "TNR" = tnrAdaptive[[iList]],
                                   "MCC" = mccAdaptive[[iList]])
}

plotframe[[(length(nObservationsVector) + 1)]] <- data.frame("nObservations" = nObservationsVector,
                             "TPR" = tprLasso,
                             "TNR" = tnrLasso,
                             "MCC" = mccLasso)


for (iList in 1:length(plotframe)) {
  plotframe[[iList]] <- plotframe[[iList]][complete.cases(plotframe[[iList]]), ]
}


for (iList in 1:(length(plotframe) - 1)) {
  ggplo <- ggplot(data = plotframe[[iList]],
                  aes(x = Gamma1)) +
    geom_point(aes(y = plotframe[[iList]]$TPR, color = cbPalette[3])) +
    geom_point(aes(y = plotframe[[iList]]$TNR, color = cbPalette[4])) + 
    geom_point(aes(y = plotframe[[iList]]$MCC, color = cbPalette[5])) + 
    geom_line(y = plotframe[[iList]]$TPR, colour = cbPalette[3], lwd = 1.5) +
    geom_line(y = plotframe[[iList]]$TNR, colour = cbPalette[4], lwd = 1.5) +
    geom_line(y = plotframe[[iList]]$MCC, colour = cbPalette[5], lwd = 1.5) +
    theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    scale_x_continuous(labels = gammaAdaptive, breaks = gammaAdaptive) + 
    labs(x = "Gamma Value",y = "Evaluation Metrics", title = paste("Evaluation Metrics for Adaptive Lasso, N Observations = ",
                                                             nObservationsVector[iList]), color = "Evaluation Metric") + 
    theme(legend.position = "top") + 
    scale_color_manual(labels = c("TPR","TNR","MCC"), values = cbPalette[3:5])
  print(ggplo)
}

for (iList in length(plotframe)) {
  ggplo <- ggplot(data = plotframe[[iList]],
                  aes(x = nObservations)) +
    geom_point(aes(y = plotframe[[iList]]$TPR, color = cbPalette[3])) +
    geom_point(aes(y = plotframe[[iList]]$TNR, color = cbPalette[4])) + 
    geom_point(aes(y = plotframe[[iList]]$MCC, color = cbPalette[5])) +
    geom_line(y = plotframe[[iList]]$TPR, colour = cbPalette[3], lwd = 1.5) +
    geom_line(y = plotframe[[iList]]$TNR, colour = cbPalette[4], lwd = 1.5) + 
    geom_line(y = plotframe[[iList]]$MCC, colour = cbPalette[5], lwd = 1.5) +
    theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    scale_x_continuous(labels = nObservationsVector, breaks = nObservationsVector) + 
    labs(x = "Number of Observations",y = "Evaluation Metrics", title = "Evaluation Metrics for Lasso", color = "Evaluation Metric") + 
    theme(legend.position = "top") + 
    scale_color_manual(labels = c("TPR","TNR","MCC"), values = cbPalette[3:5])
  print(ggplo)
}

tprAdaptiveG <- rep(0,length(nObservationsVector))
tnrAdaptiveG <- rep(0,length(nObservationsVector))
mccAdaptiveG <- rep(0,length(nObservationsVector))
for (iLength in 1:length(nObservationsVector)) {
  tprAdaptiveG[iLength] <- tprAdaptive[[iLength]][2]
  tnrAdaptiveG[iLength] <- tnrAdaptive[[iLength]][2]
  mccAdaptiveG[iLength] <- mccAdaptive[[iLength]][2]
}

tprAdaptiveG2 <- rep(0,length(nObservationsVector))
tnrAdaptiveG2 <- rep(0,length(nObservationsVector))
mccAdaptiveG2 <- rep(0,length(nObservationsVector))
for (iLength in 1:length(nObservationsVector)) {
  tprAdaptiveG2[iLength] <- tprAdaptive[[iLength]][5]
  tnrAdaptiveG2[iLength] <- tnrAdaptive[[iLength]][5]
  mccAdaptiveG2[iLength] <- mccAdaptive[[iLength]][5]
}


plotframe2 <- vector(mode = "list", length = 3)

for (iList in 1) {
  plotframe2[[iList]] <- data.frame("nObservations" = nObservationsVector,
                                   "TPR" = tprAdaptiveG,
                                   "TNR" = tnrAdaptiveG,
                                   "MCC" = mccAdaptiveG)
}

for (iList in 2) {
  plotframe2[[iList]] <- data.frame("nObservations" = nObservationsVector,
                                    "TPR" = tprAdaptiveG2,
                                    "TNR" = tnrAdaptiveG2,
                                    "MCC" = mccAdaptiveG2)
}

plotframe2[[3]] <- data.frame("nObservations" = nObservationsVector,
                                                             "TPR" = tprLasso,
                                                             "TNR" = tnrLasso,
                                                             "MCC" = mccLasso)


for (iList in 1:length(plotframe2)) {
  plotframe2[[iList]] <- plotframe2[[iList]][complete.cases(plotframe2[[iList]]), ]
}


for (iList in 1) {
  ggplo <- ggplot(data = plotframe2[[iList]],
                  aes(x = nObservations)) +
    geom_point(aes(y = plotframe2[[iList]]$TPR, color = cbPalette[3])) +
    geom_point(aes(y = plotframe2[[iList]]$TNR, color = cbPalette[4])) + 
    geom_point(aes(y = plotframe2[[iList]]$MCC, color = cbPalette[5])) +
    geom_line(y = plotframe2[[iList]]$TPR, colour = cbPalette[3], lwd = 1.5) +
    geom_line(y = plotframe2[[iList]]$TNR, colour = cbPalette[4], lwd = 1.5) + 
    geom_line(y = plotframe2[[iList]]$MCC, colour = cbPalette[5], lwd = 1.5) +
    theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
    scale_x_continuous(labels = nObservationsVector, breaks = nObservationsVector) + 
    labs(x = "Number of Observations",y = "Evaluation Metrics", title = "Evaluation Metrics for Adaptive Lasso, Gamma = 0.5", color = "Evaluation Metric") + 
    theme(legend.position = "top") + 
    scale_color_manual(labels = c("TPR","TNR","MCC"), values = cbPalette[3:5])
  print(ggplo)
}
