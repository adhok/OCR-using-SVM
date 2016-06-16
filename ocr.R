# The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital letters 
# in the English alphabet. The character images were based on 20 different fonts and each letter within these 20 fonts was randomly 
# distorted to produce a file of 20,000 unique stimuli. Each stimulus was converted into 16 primitive numerical attributes 
# (statistical moments and edge counts) which were then scaled to fit into a range of integer values from 0 through 15. 
# We typically train on the first 16000 items and then use the resulting model to predict the letter category for the remaining 4000
library(kernlab)
letters<- read.table('letterdata.txt',sep=',')
str(letters)
letters_train <- letters[1:16000,]
letters_test <- letters[16001:20000,]
letters_classifier <-ksvm(V1~.,data=letters_train,kernel="vanilladot")
str(letters_classifier)
letters_predictions<-predict(letters_classifier,letters_test)
head(letters_predictions)
#how well the classifiers performed
table(letters_predictions,letters_test$V1)
agreement_vanilla <- letters_predictions == letters_test$V1
table(agreement_vanilla)
prop.table(table(agreement_vanilla))
vanilla <- sum(agreement_vanilla==TRUE)
#improving model performance
#Gaussian Kernel is used
letter_classifier_rbf <- ksvm(V1~.,data=letters_train,kernel='rbfdot')
letter_predictions_rbf <-predict(letter_classifier_rbf,letters_test)
agreement_rbf <- letter_predictions_rbf == letters_test$V1
table(agreement_rbf)
prop.table(table(agreement_rbf))
rbf <- sum(agreement_rbf==TRUE)


#tanhdot
# letter_classifier_tan <- ksvm(V1~.,data=letters_train,kernel='tanhdot')
# letter_predictions_tan <-predict(letter_classifier_tan,letters_test)
# agreement_tan <- letter_predictions_tan == letters_test$V1
# table(agreement_tan)
# prop.table(table(agreement_tan))
#bessel
letter_classifier_bessel <- ksvm(V1~.,data=letters_train,kernel='besseldot')
letter_predictions_bessel <-predict(letter_classifier_bessel,letters_test)
agreement_bessel <- letter_predictions_rbf == letters_test$V1
table(agreement_bessel)
prop.table(table(agreement_bessel))
bessel <- sum(agreement_bessel==TRUE)
#anova
# letter_classifier_anova <- ksvm(V1~.,data=letters_train,kernel='anovadot')
# letter_predictions_anova <-predict(letter_classifier_anova,letters_test)
# agreement_anova <- letter_predictions_anova == letters_test$V1
# table(agreement_anova)
# prop.table(table(agreement_anova))
#laplace
letter_classifier_laplace <- ksvm(V1~.,data=letters_train,kernel='laplacedot')
letter_predictions_laplace <-predict(letter_classifier_laplace,letters_test)
agreement_laplace <- letter_predictions_laplace == letters_test$V1
table(agreement_laplace)
prop.table(table(agreement_laplace))
laplace <- sum(agreement_laplace == TRUE)
names <- c('Vanilla Dot','Radial Basis Function','Bessel Function','Laplace Function')
values <- c(vanilla, rbf,bessel,laplace)
df <- data.frame(names,values)
png('svm_ocr_plot.png')
barplot(df$values , names.arg=df$names,width=1,cex.names=0.5,main=paste('Classification Accuracy'))
dev.off()
library(ipred)
library(caret)
# set.seed(233)#reproducability
# ctrl <- trainControl(method = "cv", number = 100)
# bagctrl <- bagControl(fit = svmBag$fit, 
#                       predict = svmBag$pred,
#                       aggregate = svmBag$aggregate)
# svmbag <- train(V1~ ., data = letters_train, "bag", trControl = ctrl, bagControl = bagctrl)
#ctrl <- trainControl(method = 'cv',savePred= T , classProbs = T)
#mod <- train(V1~.,data=letters,method='svmLinear', trControl = ctrl)
library(randomForest)
set.seed(200)
rf <- randomForest(V1~.,data = letters_train)
p <- predict(rf,letters_test,type='response')
mean(p==letters_test$V1)




