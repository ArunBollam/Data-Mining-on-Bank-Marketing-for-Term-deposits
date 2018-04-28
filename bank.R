#reading the data
bank <- read.csv("bank-additional-full.csv")
names(bank)
bank$y <- ifelse(bank$y == "yes",1,0)

#dividing the data into training and testing data
set.seed(9)
trainingindex <- sample(nrow(bank), 0.7*nrow(bank))
bank1 <- bank[trainingindex,]
bank2 <- bank[-trainingindex,]
nrow(bank1)
nrow(bank2)

#logistic regression
logit <- glm(data = bank1, y ~.-duration,family = "binomial")

#backward selection loigistic regression model
logit1 <- step(logit, direction = "backward")
summary(logit1)

#testing the logit1 model on the test data
bankp <- predict(logit1,bank2)




install.packages("SDMTools")
install.packages("Caret")
install.packages("boot")

library(SDMTools)
library(caret)
library(boot)

#building confusion matrix
matrix1 <- confusion.matrix(bank2$y, bankp)
matrix1

#Accuracy Measures
AccuMeasures1 <- accuracy(bank2$y, bankp)
AccuMeasures1

# creating a range of values to test for accuracy
thresh=seq(0,1,by=0.05)

# Initializing a 1*20 matrix of zeros to save values of accuracy
acc = matrix(0,1,20)

# computing accuracy for different threshold values from 0 to 1 step by 0.05
for (i in 1:21){
  matrix = confusion.matrix(bank2$y,bankp,threshold=thresh[i])
  acc[i]=(matrix[1,1]+matrix[2,2])/nrow(bank2)
}
# print and plot the accuracy vs cutoff threshold values
print(c(accuracy= acc, cutoff = thresh))
plot(thresh,acc,type="l",xlab="Threshold",ylab="Accuracy", main="Validation Accuracy for Different Threshold Values")

library(ROCR)
# create the ROC, LIFT and GAIN Curves
# install.packages("ROCR)
library(ROCR)
logit_scores <- prediction(predictions=bankp, labels=bank2$y)

#PLOT ROC CURVE
logit_perf <- performance(logit_scores, "tpr", "fpr")

plot(logit_perf,
     main="ROC Curves",
     xlab="1 - Specificity: False Positive Rate",
     ylab="Sensitivity: True Positive Rate",
     col="darkblue",  lwd = 3)
abline(0,1, lty = 300, col = "green",  lwd = 3)
grid(col="aquamarine")


# AREA UNDER THE CURVE
logit_auc <- performance(logit_scores, "auc")
as.numeric(logit_auc@y.values)  ##AUC Value

#subsetting dataset


###############################################################################################

# Getting Lift Charts in R
# For getting Lift Chart in R, use measure="lift", x.measure="rpp" in the performance function.
# Get data for ROC curve and create lift chart values
logit_lift <- performance(logit_scores, measure="lift", x.measure="rpp")
plot(logit_lift,
     main="Lift Chart",
     xlab="% Populations (Percentile)",
     ylab="Lift",
     col="darkblue", lwd = 3)
abline(1,0,col="red",  lwd = 3)
grid(col="aquamarine")

##############################################################################################

## GAINS TABLE
install.packages("gains")
library(gains)
# gains table
# Three most important parameters in this functions are 
# (1) actual vector with observed target values, 
# (2) predicted vector with predicted probabilities and 
# (3) groups to mention number of groups to be created.
gains.cross <- gains(actual=bank2$y , predicted=bankp, groups=10)
print(gains.cross)

#####################################################################################################
#################         DECISION TREE         ########################
banktree <- tree(data = bank1, y ~.-duration)
banktree
plot(banktree)
text(banktree, pretty = 0)

bankt <- predict(banktree,bank2,type ="class")

cv.banktree <- cv.tree(banktree, FUN = prune.tree)
names(cv.banktree)

par(mfrow=c(1,2))
plot(cv.banktree$size,cv.banktree$dev,type="b")
plot(cv.banktree$k,cv.banktree$dev,type="b")



#######################################################
#using normalised data
bankno <- bank[bank$y == 0,]
nrow(bankno)
bankno <- bankno[sample(nrow(bankno), 0.2*nrow(bankno), replace = FALSE),]
nrow(bankno)

bankyes <- bank[bank$y == 1,]
nrow(bankyes)

bank <- rbind(bankno, bankyes)