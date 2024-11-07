
###########################################
###########################################
#
#  EXPERIMENTAL PHASE (Section 5)
#
# Tuning random forest with caret
#
# Data set: Parkinson
#
#
#
#
########################################
############   train function, from caret library
############
############ This function sets up a grid of tuning parameters for a number of 
############ classification and regression routines, fits each model and 
############ calculates a resampling based performance measure.
############ Uses "trainControl" argument from caret
###########
########################################


source("mat_square.R")

library(arules)   # for "discretize".
library(doParallel)
registerDoParallel(cores=6)

parkinsons <- read.csv("parkinsons_updrs.data", header=FALSE)
View(parkinsons)
str(parkinsons)

parkinson<-parkinsons[-1,] # first row are column names

### objective: predict "V5: motor_UPDRS" and "V6: total_UPDRS" scores from the 16 voice measures + sex + age + test_time
### The unified (total)Parkinson’s disease rating scale (UPDRS) reflects the presence and severity of symptoms 
### (but does not quantify their underlying causes). 

### From: "Accurate Telemonitoring of Parkinson’s Disease Progression by Noninvasive Speech Tests" 
### Athanasios Tsanas, Max A. Little, Member, IEEE, Patrick E. McSharry, Senior Member, IEEE, and Lorraine O. Ramig
### IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 57, NO. 4, APRIL 2010


# V1 = patient identification

parkinson<-as.data.frame(lapply(parkinson,as.numeric))

features<-c(7:22)

#### FIRST: OUTPUT VARIABLE TO PREDICT: V5: motor_UPDRS 

#arules::discretize(parkinson$V5, method = "frequency", breaks=5)

cut.points<-c(0,13,18,24,29,1000)

V5.bin <- arules::discretize(parkinson$V5, method = "fixed", breaks=cut.points, infinity=TRUE)
table(V5.bin)
parkinson<-as.data.frame(cbind(parkinson,V5.bin))


levels.V5.int<-names(table(parkinson$V5.bin))
levels(parkinson$V5.bin)<-c("<13","[13,18)","[18,24)","[24,29)",">=29")
levels.V5.ordinal.encod<-sort(unique(as.numeric(parkinson$V5.bin)))

par(mfrow = c(1, 1))
plot(parkinson$V5.bin,parkinson$V5)

parkinson$V5.bin.num<-as.numeric(parkinson$V5.bin)
table(parkinson$V5.bin)
table(parkinson$V5.bin.num)

parkinson$V5.bin.num.factor<-as.factor(parkinson$V5.bin.num)   # classes in number but factor type 

################################################################################
####### Error functions for summaryFunction argument, trainControl function
#######


standard.MAE.ord.parkinson<-function(data,lev=levels.V5.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<- mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
  value<-SMAE(Conf.mat)
  c(MAE.ord=value)
}

#########
# intervals endpoints (assume the last one is 35)
v.35<-c(0,13,18,24,29,35)

Len.35<-leng(v.35)  # intervals lengths

standard.MAE.int.35.parkinson<-function(data,lev=levels.V5.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.35)
c(MAE.int.35 = value)
}

#########
v.40<-c(0,13,18,24,29,40) # intervals endpoints (assume the last one is 40)

Len.40<-leng(v.40)  # intervals lengths

standard.MAE.int.40.parkinson<-function(data,lev=levels.V5.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.40)
c(MAE.int.40 = value)
}

#
#########
v.45<-c(0,13,18,24,29,45) # intervals endpoints (assume the last one is 45)

Len.45<-leng(v.45)  # intervals lengths

standard.MAE.int.45.parkinson<-function(data,lev=levels.V5.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.45)
c(MAE.int.45 = value)
}


#
#########
v.50<-c(0,13,18,24,29,50) # intervals endpoints (assume the last one is 50)

Len.50<-leng(v.50)  # intervals lengths

standard.MAE.int.50.parkinson<-function(data,lev=levels.V5.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.50)
c(MAE.int.50 = value)
}


#
#########
v.60<-c(0,13,18,24,29,60) # intervals endpoints (assume the last one is 60)

Len.60<-leng(v.60)  # intervals lengths

standard.MAE.int.60.parkinson<-function(data,lev=levels.V5.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.60)
c(MAE.int.60 = value)
}


#
#

################################################################################
####### preparing for k-fold cross-validation with k=10
#######

N=dim(parkinson)[1]
n=round(N/10)

set.seed(12345)
fold<-sample(c(1:10),N,replace=TRUE)
table(fold)

training<-list()
test<-list()
sub.train<-list()

for (i in 1:10)
{test[[i]]<-parkinson[which(fold==i),]
training[[i]]<-parkinson[-which(fold==i),]}


################################################################################
################################################################################
########## caret::train. Resampling method: cross-validation
################################################################################
library(caret)

ntree<-3

fitControl.Accuracy <- trainControl(
  method = "cv",
  number = 3,
  search="random")



fitControl.MAE.parkinson <- trainControl(
                           method = "cv",
                           number = 3,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = standard.MAE.ord.parkinson)


fitControl.MAE.int.35.parkinson <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.35.parkinson)


fitControl.MAE.int.40.parkinson <- trainControl(
                              method = "cv",
                              number = 3,
                               ## Evaluate performance using 
                               ## the following function
                               summaryFunction = standard.MAE.int.40.parkinson)


fitControl.MAE.int.45.parkinson <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.45.parkinson)


fitControl.MAE.int.50.parkinson <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.50.parkinson)

fitControl.MAE.int.60.parkinson <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.60.parkinson)

tuned.rf.caret.Accuracy.parkinson<-list()
tuned.rf.caret.MAE.parkinson<-list()
tuned.rf.caret.MAE.int.35.parkinson<-list()
tuned.rf.caret.MAE.int.40.parkinson<-list()
tuned.rf.caret.MAE.int.45.parkinson<-list()
tuned.rf.caret.MAE.int.50.parkinson<-list()
tuned.rf.caret.MAE.int.60.parkinson<-list()


for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.Accuracy.parkinson[[i]] <- caret::train(training[[i]][ ,features], 
                                                 training[[i]][ ,25], 
                                                 method="rf", 
                                                 metric="Accuracy",
                                                 tuneLength=10,
                                                 trControl=fitControl.Accuracy)
  
  print(i)}



########

Conf.mat.tuned.rf.caret.Accuracy.parkinson<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]]<-
  table(predict(tuned.rf.caret.Accuracy.parkinson[[i]],test[[i]][,features]),
        test[[i]][,25])
print(i)
}

##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.parkinson[[i]] <- caret::train(training[[i]][ ,features], 
                                               training[[i]][ ,25], 
                                               method="rf",
                                          metric="MAE.ord",
                                          maximize=FALSE,
                                               tuneLength=10,
                                               trControl=fitControl.MAE.parkinson)
  
  print(i)}



########

Conf.mat.tuned.rf.caret.MAE.parkinson<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.parkinson[[i]]<-
  table(predict(tuned.rf.caret.MAE.parkinson[[i]],test[[i]][,features]),
        test[[i]][,25])
print(i)
}


##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.35.parkinson[[i]] <- caret::train(training[[i]][ ,features], 
                                          training[[i]][ ,25], 
                                          method="rf",
                                          metric="MAE.int.35",
                                          maximize=FALSE,
                                          tuneLength=10,
                                          trControl=fitControl.MAE.int.35.parkinson)
  
  print(i)}



########

Conf.mat.tuned.rf.caret.MAE.int.35.parkinson<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.35.parkinson[[i]],test[[i]][,features]),
        test[[i]][,25])
print(i)
}


##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.40.parkinson[[i]] <- caret::train(training[[i]][ ,features], 
                                                           training[[i]][ ,25], 
                                                           method="rf", 
                                                           metric="MAE.int.40",
                                                           maximize=FALSE,
                                                           tuneLength=10,
                                                           trControl=fitControl.MAE.int.40.parkinson)
  
  print(i)}


########

Conf.mat.tuned.rf.caret.MAE.int.40.parkinson<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.40.parkinson[[i]],test[[i]][,features]),
        test[[i]][,25])
print(i)
}


##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.45.parkinson[[i]] <- caret::train(training[[i]][ ,features], 
                                                           training[[i]][ ,25], 
                                                           method="rf", 
                                                           metric="MAE.int.45",
                                                           maximize=FALSE,
                                                           tuneLength=10,
                                                           trControl=fitControl.MAE.int.45.parkinson)
  
  print(i)}


########

Conf.mat.tuned.rf.caret.MAE.int.45.parkinson<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.45.parkinson[[i]],test[[i]][,features]),
        test[[i]][,25])
print(i)
}


##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.50.parkinson[[i]] <- caret::train(training[[i]][ ,features], 
                                                 training[[i]][ ,25], 
                                                 method="rf", 
                                                 metric="MAE.int.50",
                                                 maximize=FALSE,
                                                 tuneLength=10,
                                                 trControl=fitControl.MAE.int.50.parkinson)
  
  print(i)}


########

Conf.mat.tuned.rf.caret.MAE.int.50.parkinson<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.50.parkinson[[i]],test[[i]][,features]),
        test[[i]][,25])
print(i)
}


##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.60.parkinson[[i]] <- caret::train(training[[i]][ ,features], 
                                                 training[[i]][ ,25], 
                                                 method="rf", 
                                                 metric="MAE.int.60",
                                                 maximize=FALSE,
                                                 tuneLength=10,
                                                 trControl=fitControl.MAE.int.60.parkinson)
  
  print(i)}



########

Conf.mat.tuned.rf.caret.MAE.int.60.parkinson<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.60.parkinson[[i]],test[[i]][,features]),
        test[[i]][,25])
print(i)
}

###########


################################################################################
################################################################################
################## Accuracy, SMAE and SMAE.int
################################################################################
#
Accuracy.M.Accuracy.parkinson<-vector()
Accuracy.M.MAE.parkinson<-vector()
Accuracy.M.MAE.int.35.parkinson<-vector() 
Accuracy.M.MAE.int.40.parkinson<-vector() 
Accuracy.M.MAE.int.45.parkinson<-vector() 
Accuracy.M.MAE.int.50.parkinson<-vector() 
Accuracy.M.MAE.int.60.parkinson<-vector() 
#
SMAE.M.Accuracy.parkinson<-vector()
SMAE.M.MAE.parkinson<-vector()
SMAE.M.MAE.int.35.parkinson<-vector()
SMAE.M.MAE.int.40.parkinson<-vector()
SMAE.M.MAE.int.45.parkinson<-vector()
SMAE.M.MAE.int.50.parkinson<-vector()
SMAE.M.MAE.int.60.parkinson<-vector()
#
SMAE.int.35.M.Accuracy.parkinson<-vector()
SMAE.int.35.M.MAE.parkinson<-vector()
SMAE.int.35.M.MAE.int.35.parkinson<-vector()
SMAE.int.35.M.MAE.int.40.parkinson<-vector()
SMAE.int.35.M.MAE.int.45.parkinson<-vector()
SMAE.int.35.M.MAE.int.50.parkinson<-vector()
SMAE.int.35.M.MAE.int.60.parkinson<-vector()
#
SMAE.int.40.M.Accuracy.parkinson<-vector()
SMAE.int.40.M.MAE.parkinson<-vector()
SMAE.int.40.M.MAE.int.35.parkinson<-vector()
SMAE.int.40.M.MAE.int.40.parkinson<-vector()
SMAE.int.40.M.MAE.int.45.parkinson<-vector()
SMAE.int.40.M.MAE.int.50.parkinson<-vector()
SMAE.int.40.M.MAE.int.60.parkinson<-vector()
#
SMAE.int.45.M.Accuracy.parkinson<-vector()
SMAE.int.45.M.MAE.parkinson<-vector()
SMAE.int.45.M.MAE.int.35.parkinson<-vector()
SMAE.int.45.M.MAE.int.40.parkinson<-vector()
SMAE.int.45.M.MAE.int.45.parkinson<-vector()
SMAE.int.45.M.MAE.int.50.parkinson<-vector()
SMAE.int.45.M.MAE.int.60.parkinson<-vector()
#
SMAE.int.50.M.Accuracy.parkinson<-vector()
SMAE.int.50.M.MAE.parkinson<-vector()
SMAE.int.50.M.MAE.int.35.parkinson<-vector()
SMAE.int.50.M.MAE.int.40.parkinson<-vector()
SMAE.int.50.M.MAE.int.45.parkinson<-vector()
SMAE.int.50.M.MAE.int.50.parkinson<-vector()
SMAE.int.50.M.MAE.int.60.parkinson<-vector()
#
SMAE.int.60.M.Accuracy.parkinson<-vector()
SMAE.int.60.M.MAE.parkinson<-vector()
SMAE.int.60.M.MAE.int.35.parkinson<-vector()
SMAE.int.60.M.MAE.int.40.parkinson<-vector()
SMAE.int.60.M.MAE.int.45.parkinson<-vector()
SMAE.int.60.M.MAE.int.50.parkinson<-vector()
SMAE.int.60.M.MAE.int.60.parkinson<-vector()



for (i in 1:10)
{
  Accuracy.M.Accuracy.parkinson[i]<-sum(diag(Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]]))/sum(Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]])
  Accuracy.M.MAE.parkinson[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.parkinson[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.parkinson[[i]])
  Accuracy.M.MAE.int.35.parkinson[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]])
  Accuracy.M.MAE.int.40.parkinson[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]])
  Accuracy.M.MAE.int.45.parkinson[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]])
  Accuracy.M.MAE.int.50.parkinson[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]])
  Accuracy.M.MAE.int.60.parkinson[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]])
  ##
  SMAE.M.Accuracy.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.35.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.40.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.45.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.50.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.60.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod))
  ##
  SMAE.int.35.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  ##
  SMAE.int.40.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  ##
  SMAE.int.45.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  ##
  SMAE.int.50.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  ##
  SMAE.int.60.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
}



###############################
###############################
###   SMAE.int

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot( 
    SMAE.int.35.M.MAE.int.35.parkinson,
    SMAE.int.40.M.MAE.int.40.parkinson,
    SMAE.int.45.M.MAE.int.45.parkinson,
    SMAE.int.50.M.MAE.int.50.parkinson,
    SMAE.int.60.M.MAE.int.60.parkinson,
    main=paste(" "), 
    xlab="length assigned to the rigthmost interval", ylab=" ",names=c("6", "11", "16",
                                                                       "21", "31"),
    col=c2,medcol=c3)
  
}

#
#
#

Group.five<-c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10))


SMAE.int.all.parkinson<-c(SMAE.int.35.M.MAE.int.35.parkinson,
                        SMAE.int.40.M.MAE.int.40.parkinson,
                        SMAE.int.45.M.MAE.int.45.parkinson,
                        SMAE.int.50.M.MAE.int.50.parkinson,
                        SMAE.int.60.M.MAE.int.60.parkinson)

p1<-shapiro.test(SMAE.int.35.M.MAE.int.35.parkinson)$p.value
p2<-shapiro.test(SMAE.int.40.M.MAE.int.40.parkinson)$p.value
p3<-shapiro.test(SMAE.int.45.M.MAE.int.45.parkinson)$p.value
p4<-shapiro.test(SMAE.int.50.M.MAE.int.50.parkinson)$p.value
p5<-shapiro.test(SMAE.int.60.M.MAE.int.60.parkinson)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05 & p5>=0.05)
{
  test.SMAE.int.less.parkinson<-pairwise.t.test(SMAE.int.all.parkinson,Group.five,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater.parkinson<-pairwise.t.test(SMAE.int.all.parkinson,Group.five,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.int.less.parkinson<-pairwise.wilcox.test(SMAE.int.all.parkinson,Group.five,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater.parkinson<-pairwise.wilcox.test(SMAE.int.all.parkinson,Group.five,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.int.less.parkinson
test.SMAE.int.greater.parkinson

# #
# #

p1<-shapiro.test(SMAE.int.35.M.Accuracy.parkinson)$p.value
p2<-shapiro.test(SMAE.int.35.M.MAE.parkinson)$p.value
p3<-shapiro.test(SMAE.int.35.M.MAE.int.35.parkinson)$p.value
p4<-shapiro.test(SMAE.int.35.M.MAE.int.40.parkinson)$p.value
p5<-shapiro.test(SMAE.int.35.M.MAE.int.45.parkinson)$p.value
p6<-shapiro.test(SMAE.int.35.M.MAE.int.50.parkinson)$p.value
p7<-shapiro.test(SMAE.int.35.M.MAE.int.60.parkinson)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.Accuracy.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.int.40.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.int.45.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.int.50.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.int.60.parkinson, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.Accuracy.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.int.40.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.int.45.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.int.50.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.parkinson,SMAE.int.35.M.MAE.int.60.parkinson, paired=TRUE, alternative="greater")$p.value

# #

p1<-shapiro.test(SMAE.int.40.M.Accuracy.parkinson)$p.value
p2<-shapiro.test(SMAE.int.40.M.MAE.parkinson)$p.value
p3<-shapiro.test(SMAE.int.40.M.MAE.int.35.parkinson)$p.value
p4<-shapiro.test(SMAE.int.40.M.MAE.int.40.parkinson)$p.value
p5<-shapiro.test(SMAE.int.40.M.MAE.int.45.parkinson)$p.value
p6<-shapiro.test(SMAE.int.40.M.MAE.int.50.parkinson)$p.value
p7<-shapiro.test(SMAE.int.40.M.MAE.int.60.parkinson)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.Accuracy.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.35.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.45.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.50.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.60.parkinson, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.Accuracy.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.35.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.45.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.50.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.60.parkinson, paired=TRUE, alternative="greater")$p.value

# #

p1<-shapiro.test(SMAE.int.45.M.Accuracy.parkinson)$p.value
p2<-shapiro.test(SMAE.int.45.M.MAE.parkinson)$p.value
p3<-shapiro.test(SMAE.int.45.M.MAE.int.35.parkinson)$p.value
p4<-shapiro.test(SMAE.int.45.M.MAE.int.40.parkinson)$p.value
p5<-shapiro.test(SMAE.int.45.M.MAE.int.45.parkinson)$p.value
p6<-shapiro.test(SMAE.int.45.M.MAE.int.50.parkinson)$p.value
p7<-shapiro.test(SMAE.int.45.M.MAE.int.60.parkinson)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.Accuracy.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.35.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.40.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.50.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.60.parkinson, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.Accuracy.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.35.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.40.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.50.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.60.parkinson, paired=TRUE, alternative="greater")$p.value

# #

p1<-shapiro.test(SMAE.int.50.M.Accuracy.parkinson)$p.value
p2<-shapiro.test(SMAE.int.50.M.MAE.parkinson)$p.value
p3<-shapiro.test(SMAE.int.50.M.MAE.int.35.parkinson)$p.value
p4<-shapiro.test(SMAE.int.50.M.MAE.int.40.parkinson)$p.value
p5<-shapiro.test(SMAE.int.50.M.MAE.int.45.parkinson)$p.value
p6<-shapiro.test(SMAE.int.50.M.MAE.int.50.parkinson)$p.value
p7<-shapiro.test(SMAE.int.50.M.MAE.int.60.parkinson)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.Accuracy.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.35.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.40.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.45.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.60.parkinson, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.Accuracy.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.35.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.40.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.45.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.60.parkinson, paired=TRUE, alternative="greater")$p.value

# #

p1<-shapiro.test(SMAE.int.60.M.Accuracy.parkinson)$p.value
p2<-shapiro.test(SMAE.int.60.M.MAE.parkinson)$p.value
p3<-shapiro.test(SMAE.int.60.M.MAE.int.35.parkinson)$p.value
p4<-shapiro.test(SMAE.int.60.M.MAE.int.40.parkinson)$p.value
p5<-shapiro.test(SMAE.int.60.M.MAE.int.45.parkinson)$p.value
p6<-shapiro.test(SMAE.int.60.M.MAE.int.50.parkinson)$p.value
p7<-shapiro.test(SMAE.int.60.M.MAE.int.60.parkinson)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.Accuracy.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.int.35.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.int.40.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.int.45.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.int.50.parkinson, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.Accuracy.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.int.35.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.int.40.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.int.45.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.60.M.MAE.int.60.parkinson,SMAE.int.60.M.MAE.int.50.parkinson, paired=TRUE, alternative="greater")$p.value


