setwd("D:/Masters Program Chalmers/Projects and Labs/SLBD/Final Project/Part 2")

library(cluster)  #pam and silhouette
library(subspace) #proClus
library(MASS)
library(tidyverse)
library(latex2exp) # Latex in ggplot2 labels
library(caret)
library(mclust) #GMM
library(readr)
library(factoextra) #Fviz
library(plotGMM)
library(randomForest)
library(Rtsne)

cbPalette <- c(
  "#999999","#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7") # colour-blind friendly palette

# Data --------------------------------------------------------------------

features <- X
rm(X)
features <- as.data.frame(features)
nObservations <- nrow(features)
nFeatures <- ncol(features)

features <- scale(x = features)

#hist(dist(features),main = "Histogram of Euclidean distances") 

# PCA ---------------------------------------------------------------------

pca <- prcomp(features,scale = TRUE)
plot(pca$x[,1],pca$x[,2], xlab = "PC1",ylab = "PC2",main = "Projection of observations on PC1 and PC2", col = cbPalette[7])
plot(pca$sdev,xlab = "Number of features",ylab = "PCA Standard deviation", main = "Average number of principal components to explaining variability") #Suggests around 50 dimensions

pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100,1)
barplot(pca.var.per,  ylab="Explained variance",main = "Bar plot to represent PC explaining variability",
        xlab = "Principal Components", col=cbPalette[1])

loadingScores <- pca$rotation[,1]  #PC1
variableScores <- abs(loadingScores)
varScoresRanked <- sort(variableScores,decreasing = T)
variablesRF <- order(variableScores,decreasing = T)

# Silhouette width --------------------------------------------------------

# Average silhouette width
Kmax <- 32
avg_sws <- sapply(2:Kmax, function(K) {
  mean(cluster::silhouette(
    cluster::pam(features, K))[,3])
})

data_plot <- tibble(
  K = 2:Kmax,
  avg_sw = avg_sws)

p2 <- ggplot(data_plot) +
  geom_line(aes(x = K, y = avg_sw)) +
  scale_y_continuous("Average Silhouette Width") +
  scale_x_continuous(breaks = data_plot$K,labels = data_plot$K) +
  labs(x = "Number of clusters",y = "Average Silhouette Width", title = "Silhouette width with varying number of clusters") + 
  theme(legend.position = "top") + 
  theme_minimal() +
  theme(
    axis.text = element_text(size = 9),
    axis.title = element_text(size = 9))

print(p2)

#Suggest that optimal clusters are 2-4


# Methods -----------------------------------------------------------------

# Plotting for all clusters best method by BIC ----------------------------

gmmClustering <- Mclust(features, )
plot(gmmClustering,what = "BIC",legendArgs = list(x = "bottomright"),xlab = "Number of clusters"
     ,ylab = "BIC Value", main = "BIC on different clusters")

BICSummary <- summary(gmmClustering$BIC)
Method <- substring(names(BICSummary[1]), first=1, last=3)
#gmmClustering <- Mclust(features, modelNames = "Method",)
gmmClustering <- Mclust(features, modelNames = "VVI",G = 4)

fviz_mclust(object = gmmClustering,what = "classification", geom = "point"
            ,ggtheme = theme_classic(),legend = "right"
            ,main = paste("GMM Clusters for ", Method),xlab = "PC1", ylab = "PC2")


nClusters <- 4 #From BICSummary


# Random forest to get average dimensionality -----------------------------

observationLabel <- gmmClustering[["classification"]]
observationLabel <- as.factor(observationLabel)
nTree <- 200
randomForestObject <- randomForest(observationLabel~.,data = features, na.action = na.omit,
  importance = TRUE, ntree = nTree)

varImportance <- varImpPlot(randomForestObject, sort=TRUE, n.var=nrow(randomForestObject$importance),
           type=NULL, class=NULL, scale=TRUE, labels = "", main = "Variable Importance Plot")

avgDimension <- length(which(varImportance[,2] > 2))
avgDimension <- avgDimension + length(which(varImportance[,1] > 4))
avgDimension <- floor(avgDimension/2)

avgDimension

# tSNE --------------------------------------------------------------------

perplexity <- floor(sqrt(nObservations))
par(mfrow=c(2,2))
for(iter in c(200, 300, 500,1000)) {
  tSNE <- Rtsne(t(log10(features+abs(min(features))+1)), 
                perplexity=perplexity, max_iter=iter)
  plot(tSNE$Y, col="orange", xlab="tSNE1", ylab="tSNE2")
  mtext(paste0("Iterations = ", iter))
}

 # Supspace ----------------------------------------------------------------

nClusters <- seq(from = 3, to = 12, by = 1)
dimensionVector <- avgDimension

datapointsClustered <- vector(mode = "list",length = length(nClusters))
influentialFeatures <- vector(mode = "list",length = length(nClusters))

for (iClusterList in 1:length(nClusters)) {
  datapointsClustered[[iClusterList]] <- rep(0,times = nObservations)
  influentialFeatures[[iClusterList]] <- rep(0,times = nFeatures)
}

nMaxRuns <- 50
featureHits <- matrix(0,nrow = nFeatures,ncol = nMaxRuns)

for (iClusterList in 1:length(nClusters)) {
  iRun <- 1
  while (iRun <= nMaxRuns) {
    proclusFit <- ProClus(features, k = nClusters[iClusterList], d = dimensionVector)
    for (iCluster in 1:nClusters[iClusterList]) {
      datapointsClustered[[iClusterList]][proclusFit[[iCluster]][["objects"]]] <- 
        datapointsClustered[[iClusterList]][proclusFit[[iCluster]][["objects"]]] + 1
      influentialFeatures[[iClusterList]][which(proclusFit[[iCluster]][["subspace"]] == T)] <-
        influentialFeatures[[iClusterList]][which(proclusFit[[iCluster]][["subspace"]] == T)] + 1
    }
    iRun <- iRun + 1
  }
}


influenceF <- rep(0,times = nFeatures)
influenceD <- rep(0,times = nObservations)
for (iFeature in 1:length(influentialFeatures)) {
  influenceF <- influenceF + influentialFeatures[[iFeature]]
  influenceD <- influenceD + datapointsClustered[[iFeature]]
}
hist(influenceD)
hist(influenceF)

h <- hist(influenceF,main = "Histogram of Feature Hits",xlab = "Number of Hits",col = "gray", ylab = "Frequency of Feature")
text(h$mids,h$counts,labels=h$counts, adj=c(0.5, -0.2))

d <- hist(influenceD/50,main = "Histogram of Observation Clustering Index",ylab = "Frequency of Observation",xlab = "Observation Clustering Index",col = "gray")
text(d$mids,d$counts,labels=d$counts, adj=c(0.5, -0.2))

highinfluencefeatures <- which(influenceF > 1275)
observationsOutlier <- which(influenceD/50 <= 1)


correlationMatrix <- cor(features[,highinfluencefeatures])
corrplot::corrplot(correlationMatrix,type = "lower")


# Average silhouette width

Kmax <- 32
avg_sws <- sapply(2:Kmax, function(K) {
  mean(cluster::silhouette(
    cluster::pam(features[-observationsOutlier,highinfluencefeatures], K))[,3])
})

data_plot <- tibble(
  K = 2:Kmax,
  avg_sw = avg_sws)

p2 <- ggplot(data_plot) +
  geom_line(aes(x = K, y = avg_sw)) +
  scale_y_continuous("Average Silhouette Width") +
  scale_x_continuous(breaks = data_plot$K,labels = data_plot$K) +
  labs(x = "Number of clusters",y = "Average Silhouette Width", title = "Silhouette width with varying number of clusters") + 
  theme(legend.position = "top") + 
  theme_minimal() +
  theme(
    axis.text = element_text(size = 9),
    axis.title = element_text(size = 9))

print(p2)

plot(hclust(dist(features[-observationsOutlier,highinfluencefeatures]),method = "average"), xlab = "Observations")
