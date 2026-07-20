data <- read.table('dermatology.dat')
summary(data)
data$class <- as.factor(data$class)
# majority class
table(data$class)/nrow(data) # predicts class 1 with 0.31 acc
library(caret)
set.seed(123)
idx <- createDataPartition(data$class,p=0.7,list=F)
trn <- data[idx,]
tst <- data[-idx,]
# verify the proportions are mantained
table(trn$class)/nrow(trn)
table(tst$class)/nrow(tst)
#knn
set.seed(123)
trn.ctrl <- trainControl(method = 'cv',number = 10)
knn.fit <- train(trn[,-21],trn$class,method = 'knn',preProcess = c('center','scale'),
                 trControl = trn.ctrl, tuneGrid = expand.grid(k=1:30))
knn.fit# shows accuracy and kappa for different k values
plot(knn.fit)# to see it visually
pred.knn <- predict(knn.fit,tst)
confusionMatrix(pred.knn,tst$class)


#tree
library(rpart)
library(rattle)
set.seed(123)
tree1 <- rpart(class~.,data = trn)# uses Gini index, parms=list(split="information")
# to use cross entropy
fancyRpartPlot(tree1)
tree1$variable.importance
# pruning with rpart
set.seed(123)
printcp(tree1)# guardo l'albero più piccolo con xerror e xstd più bassi
plotcp(tree1)# guardo l'albero più piccolo sotto la linea (scelgo cp leggermente maggiore)
tree1_pruned <- prune(tree1,cp=0.015)
fancyRpartPlot(tree1_pruned)
tree1_pruned$variable.importance
# predictions on test set, confusion matrix
pred_tree1_pruned <- predict(tree1_pruned,tst,type = 'class',na.action=na.pass)
confusionMatrix(pred_tree1_pruned,tst$class)
##nb 
library(caret)
nb.cv <- train(class~., data = trn,
               method = 'naive_bayes',
               trControl = trainControl('cv',number = 10),
               na.action = na.pass)
nb.cv # per vedere se ha usato o no i kernel
plot(nb.cv$finalModel) #vedere le density (come le variabili distinguono le classi)
pred.nb.cv <- predict(nb.cv,tst,na.action = na.pass)# devo specificare cosa fare
# con gli na
confusionMatrix(pred.nb.cv,tst$class)

#LDA
set.seed(123)
library(MASS)
lda_fit <- lda(class ~ ., data = trn)
pred_lda <- predict(lda_fit, newdata = tst)
pred_lda$class
pred_lda$posterior # posterior probabilities
confusionMatrix(pred_lda$class,tst$class) 
lda_fit$scaling #coefficiants of variables in discriminant direction
lda_fit$svd #sqrt(eigenvalue) (quantity distinction explained)
lda_fit$svd^2 / sum(lda_fit$svd^2) # here proportion of distinction explained
# project training data and tst data onto discriminant axis
pred_trn <- predict(lda_fit, trn)
pred_tst <- predict(lda_fit, tst)
scores_train <- pred_trn$x  
scores_test  <- pred_tst$x
#plots
#true classes on tst
plot(LD2~LD1, data=scores_test,
     col = as.integer(tst$class),
     pch = 20,
     xlab = "LD1", ylab = "LD2",
     main = "LDA scores by true class (trn)"
     )
legend(x=5, y=-5, legend=c("1","2","3","4","5","6"), col=1:6, pch=19)
plot(LD3~LD2, data=scores_test,
     col = as.integer(tst$class),
     pch = 20,
     xlab = "LD2", ylab = "LD3",
     main = "LDA scores by true class (trn)" 
     )
legend(x=-18, y=0, legend=c("1","2","3","4","5","6"), col=1:6, pch=19)

