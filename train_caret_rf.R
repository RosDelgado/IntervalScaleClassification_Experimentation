###########################################
###########################################
#
#  EXPERIMENTAL PHASE (Section 5)
#
# Tuning random forest with caret
#
# Data set: face_age
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

library(arules)   # for "discretize" function
library(doParallel)
registerDoParallel(cores=6)

load("faces.grey.32.Rda")  # load dataframe "db.faces.grey.32"
df<-db.faces.grey.32
str(df)


# #####  variable "age": binning to "age.bin", with 5 intervals
# 

cut.points<-c(0,2,10,15,35,60,1000)

age.bin <- arules::discretize(df$age, method = "fixed", breaks=cut.points, infinity=TRUE)
table(age.bin)
df<-as.data.frame(cbind(df,age.bin))

df$age.bin<-as.factor(df$age.bin)

levels.age.int<-names(table(df$age.bin))
levels(df$age.bin)<-c("<2","[2,10)","[10,15)","[15,35)","[35,60)",">=60")
levels.age.ordinal.encod<-unique(as.numeric(df$age.bin))

par(mfrow = c(1, 1))
plot(df$age.bin,df$age)


levels.age.ordinal.encod<-unique(as.numeric(df$age.bin))
df$age.bin.num<-as.numeric(df$age.bin)
table(df$age.bin)

df$age.bin.num.factor<-as.factor(df$age.bin.num)   # we need classes as number but factor 

################################################################################
####### Error functions for summaryFunction argument, trainControl function
#######

standard.MAE.ord<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<- mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
  value<-SMAE(Conf.mat)
  c(MAE.ord=value)
}

#########
v.80<-c(0,2,10,15,35,60,80) # intervals endpoints (assume the last one is 80)

Len.80<-leng(v.80)  # intervals lengths

standard.MAE.int.80<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.80)
c(MAE.int.80 = value)
}

#########
v.90<-c(0,2,10,15,35,60,90) # intervals endpoints (assume the last one is 90)

Len.90<-leng(v.90)  # intervals lengths

standard.MAE.int.90<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.90)
c(MAE.int.90 = value)
}


#########
v.100<-c(0,2,10,15,35,60,100) # intervals endpoints (assume the last one is 100)

Len.100<-leng(v.100)  # intervals lengths

standard.MAE.int.100<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.100)
c(MAE.int.100 = value)
}

#########
v.110<-c(0,2,10,15,35,60,110) # intervals endpoints (assume the last one is 110)

Len.110<-leng(v.110)  # intervals lengths

standard.MAE.int.110<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.110)
c(MAE.int.110 = value)
}

#########
v.120<-c(0,2,10,15,35,60,120) # intervals endpoints (assume the last one is 120)

Len.120<-leng(v.120)  # intervals lengths

standard.MAE.int.120<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.120)
c(MAE.int.120 = value)
}

#
#

################################################################################
####### preparing for k-fold cross-validation with k=5
#######

N=dim(df)[1]
n=round(N/10)

set.seed(12345)
fold<-sample(c(1:10),N,replace=TRUE)
table(fold)

training<-list()
test<-list()
sub.train<-list()
#sub.test<-list()

for (i in 1:10)
{test[[i]]<-df[which(fold==i),]
training[[i]]<-df[-which(fold==i),]}

for (i in 1:10)
{set.seed(12345)
  random.sampl<-sample(which(fold!=i),2000,replace=FALSE)
  sub.train[[i]]<-df[random.sampl,]}

################################################################################
################################################################################
########## caret::train. Resampling method: cross-validation
################################################################################
library(caret)

mtry<-sqrt(ncol(df)-4)
ntree<-3

fitControl.Accuracy <- trainControl(
  method = "cv",
  number = 3,
  search="random")

fitControl.MAE <- trainControl(
                           method = "cv",
                           number = 3,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = standard.MAE.ord)

fitControl.MAE.int.80 <- trainControl(
                              method = "cv",
                              number = 3,
                               ## Evaluate performance using 
                               ## the following function
                               summaryFunction = standard.MAE.int.80)

fitControl.MAE.int.90 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.90)

fitControl.MAE.int.100 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.100)

fitControl.MAE.int.110 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.110)

fitControl.MAE.int.120 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.120)


tuned.rf.caret.Accuracy<-list()
tuned.rf.caret.MAE<-list()
tuned.rf.caret.MAE.int.80<-list()
tuned.rf.caret.MAE.int.90<-list()
tuned.rf.caret.MAE.int.100<-list()
tuned.rf.caret.MAE.int.110<-list()
tuned.rf.caret.MAE.int.120<-list()

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.Accuracy[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf", 
                                                 metric="Accuracy",
                                                 tuneLength=10,
                                                 trControl=fitControl.Accuracy)
  
  print(i)}



########

Conf.mat.tuned.rf.caret.Accuracy<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.Accuracy[[i]]<-
  table(predict(tuned.rf.caret.Accuracy[[i]],test[[i]]),
        test[[i]][,1028])
print(i)
}

##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                               sub.train[[i]][ ,1028], 
                                               method="rf",
                                          metric="MAE.ord",
                                          maximize=FALSE,
                                               tuneLength=10,
                                               trControl=fitControl.MAE)
  
  print(i)}



########

Conf.mat.tuned.rf.caret.MAE<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE[[i]]<-
  table(predict(tuned.rf.caret.MAE[[i]],test[[i]]),
        test[[i]][,1028])
print(i)
}


##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.80[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                          sub.train[[i]][ ,1028], 
                                          method="rf", 
                                          metric="MAE.int.80",
                                          maximize=FALSE,
                                          tuneLength=10,
                                          trControl=fitControl.MAE.int.80)
  
  print(i)}



########

Conf.mat.tuned.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.80[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.80[[i]],test[[i]]),
        test[[i]][,1028])
print(i)
}


##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.90[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf",
                                                 metric="MAE.int.90",
                                                 maximize=FALSE,
                                                 tuneLength=10,
                                                 trControl=fitControl.MAE.int.90)
  
  print(i)}


########

Conf.mat.tuned.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.90[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.90[[i]],test[[i]]),
        test[[i]][,1028])
print(i)
}



##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.100[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf", 
                                                 metric="MAE.int.100",
                                                 maximize=FALSE,
                                                 tuneLength=10,
                                                 trControl=fitControl.MAE.int.100)
  
  print(i)}


########

Conf.mat.tuned.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.100[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.100[[i]],test[[i]]),
        test[[i]][,1028])
print(i)
}


##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.110[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf", 
                                                 metric="MAE.int.110",
                                                 maximize=FALSE,
                                                 tuneLength=10,
                                                 trControl=fitControl.MAE.int.110)
  
  print(i)}


########

Conf.mat.tuned.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.110[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.110[[i]],test[[i]]),
        test[[i]][,1028])
print(i)
}



##############

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.120[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf", 
                                                 metric="MAE.int.120",
                                                 maximize=FALSE,
                                                 tuneLength=10,
                                                 trControl=fitControl.MAE.int.120)
  
  print(i)}



########

Conf.mat.tuned.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{Conf.mat.tuned.rf.caret.MAE.int.120[[i]]<-
  table(predict(tuned.rf.caret.MAE.int.120[[i]],test[[i]]),
        test[[i]][,1028])
print(i)
}


################################################################################
################################################################################
################## Confusion matrices, Accuracy, SMAE and SMAE.int
################################################################################
#
Accuracy.M.Accuracy<-vector()
Accuracy.M.MAE<-vector()
Accuracy.M.MAE.int.80<-vector() 
Accuracy.M.MAE.int.90<-vector() 
Accuracy.M.MAE.int.100<-vector() 
Accuracy.M.MAE.int.110<-vector() 
Accuracy.M.MAE.int.120<-vector() 
#
SMAE.M.Accuracy<-vector()
SMAE.M.MAE<-vector()
SMAE.M.MAE.int.80<-vector()
SMAE.M.MAE.int.90<-vector()
SMAE.M.MAE.int.100<-vector()
SMAE.M.MAE.int.110<-vector()
SMAE.M.MAE.int.120<-vector()
#
SMAE.int.80.M.Accuracy<-vector()
SMAE.int.80.M.MAE<-vector()
SMAE.int.80.M.MAE.int.80<-vector()
#
SMAE.int.90.M.Accuracy<-vector()
SMAE.int.90.M.MAE<-vector()
SMAE.int.90.M.MAE.int.90<-vector()
#
SMAE.int.100.M.Accuracy<-vector()
SMAE.int.100.M.MAE<-vector()
SMAE.int.100.M.MAE.int.100<-vector()
#
SMAE.int.110.M.Accuracy<-vector()
SMAE.int.110.M.MAE<-vector()
SMAE.int.110.M.MAE.int.110<-vector()
#
SMAE.int.120.M.Accuracy<-vector()
SMAE.int.120.M.MAE<-vector()
SMAE.int.120.M.MAE.int.120<-vector()

for (i in 1:10)
{
  Accuracy.M.Accuracy[i]<-sum(diag(Conf.mat.tuned.rf.caret.Accuracy[[i]]))/sum(Conf.mat.tuned.rf.caret.Accuracy[[i]])
  Accuracy.M.MAE[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE[[i]])
  Accuracy.M.MAE.int.80[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.80[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.80[[i]])
  Accuracy.M.MAE.int.90[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.90[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.90[[i]])
  Accuracy.M.MAE.int.100[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.100[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.100[[i]])
  Accuracy.M.MAE.int.110[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.110[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.110[[i]])  
  Accuracy.M.MAE.int.120[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.120[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.120[[i]])
  ##
  SMAE.M.Accuracy[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.80[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.80[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.90[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.90[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.100[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.100[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.110[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.110[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.120[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.120[[i]],levels.age.ordinal.encod))
  ##
  SMAE.int.80.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.80)
  SMAE.int.80.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),Len.80)
  SMAE.int.80.M.MAE.int.80[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.80[[i]],levels.age.ordinal.encod),Len.80)
  ##
  SMAE.int.90.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.90)
  SMAE.int.90.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),Len.90)
  SMAE.int.90.M.MAE.int.90[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.90[[i]],levels.age.ordinal.encod),Len.90)
  ##
  SMAE.int.100.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.100)
  SMAE.int.100.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),Len.100)
  SMAE.int.100.M.MAE.int.100[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.100[[i]],levels.age.ordinal.encod),Len.100)
  ##
  SMAE.int.110.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.110)
  SMAE.int.110.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),Len.110)
  SMAE.int.110.M.MAE.int.110[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.110[[i]],levels.age.ordinal.encod),Len.110)  
  ##
  SMAE.int.120.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.120)
  SMAE.int.120.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),Len.120)
  SMAE.int.120.M.MAE.int.120[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.120[[i]],levels.age.ordinal.encod),Len.120)  
}

Accuracy.M.Accuracy
Accuracy.M.MAE
Accuracy.M.MAE.int.80
Accuracy.M.MAE.int.90
Accuracy.M.MAE.int.100
Accuracy.M.MAE.int.110
Accuracy.M.MAE.int.120

mean(Accuracy.M.Accuracy,na.rm=TRUE)
mean(Accuracy.M.MAE,na.rm=TRUE)
mean(Accuracy.M.MAE.int.80,na.rm=TRUE)
mean(Accuracy.M.MAE.int.90,na.rm=TRUE)
mean(Accuracy.M.MAE.int.100,na.rm=TRUE)
mean(Accuracy.M.MAE.int.110,na.rm=TRUE)
mean(Accuracy.M.MAE.int.120,na.rm=TRUE)


SMAE.M.Accuracy
SMAE.M.MAE
SMAE.M.MAE.int.80
SMAE.M.MAE.int.90
SMAE.M.MAE.int.100
SMAE.M.MAE.int.110
SMAE.M.MAE.int.120

mean(SMAE.M.Accuracy,na.rm=TRUE)
mean(SMAE.M.MAE,na.rm=TRUE)
mean(SMAE.M.MAE.int.80,na.rm=TRUE)
mean(SMAE.M.MAE.int.90,na.rm=TRUE)
mean(SMAE.M.MAE.int.100,na.rm=TRUE)
mean(SMAE.M.MAE.int.110,na.rm=TRUE)
mean(SMAE.M.MAE.int.120,na.rm=TRUE)

SMAE.int.80.M.Accuracy
SMAE.int.80.M.MAE
SMAE.int.80.M.MAE.int.80

mean(SMAE.int.80.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.80.M.MAE,na.rm=TRUE)
mean(SMAE.int.80.M.MAE.int.80,na.rm=TRUE)

SMAE.int.90.M.Accuracy
SMAE.int.90.M.MAE
SMAE.int.90.M.MAE.int.90

mean(SMAE.int.90.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.90.M.MAE,na.rm=TRUE)
mean(SMAE.int.90.M.MAE.int.90,na.rm=TRUE)

SMAE.int.100.M.Accuracy
SMAE.int.100.M.MAE
SMAE.int.100.M.MAE.int.100

mean(SMAE.int.100.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.100.M.MAE,na.rm=TRUE)
mean(SMAE.int.100.M.MAE.int.100,na.rm=TRUE)

SMAE.int.110.M.Accuracy
SMAE.int.110.M.MAE
SMAE.int.110.M.MAE.int.110

mean(SMAE.int.110.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.110.M.MAE,na.rm=TRUE)
mean(SMAE.int.110.M.MAE.int.110,na.rm=TRUE)

SMAE.int.120.M.Accuracy
SMAE.int.120.M.MAE
SMAE.int.120.M.MAE.int.120

mean(SMAE.int.120.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.120.M.MAE,na.rm=TRUE)
mean(SMAE.int.120.M.MAE.int.120,na.rm=TRUE)


###############################
###############################
### BOXPLOTS AND TESTS

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.int.80.M.MAE.int.80,
          SMAE.int.90.M.MAE.int.90,
          SMAE.int.100.M.MAE.int.100,
          SMAE.int.110.M.MAE.int.110,
          SMAE.int.120.M.MAE.int.120,
          main=paste(" "), 
          xlab="length assigned to the rightmost interval", ylab=" ",names=c("20", "30", "40", "50", "60"),
          col=c2,medcol=c3)
  
}

# 
# 


Group.five<-c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10))

SMAE.int.all<-c(SMAE.int.80.M.MAE.int.80,
                    SMAE.int.90.M.MAE.int.90,
                    SMAE.int.100.M.MAE.int.100,
                    SMAE.int.110.M.MAE.int.110,
                    SMAE.int.120.M.MAE.int.120)

p1<-shapiro.test(SMAE.int.80.M.MAE.int.80)$p.value
p2<-shapiro.test(SMAE.int.90.M.MAE.int.90)$p.value
p3<-shapiro.test(SMAE.int.100.M.MAE.int.100)$p.value
p4<-shapiro.test(SMAE.int.110.M.MAE.int.110)$p.value
p5<-shapiro.test(SMAE.int.120.M.MAE.int.120)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05 & p5>=0.05)
{
  test.SMAE.int.less<-pairwise.t.test(SMAE.int.all,Group.five,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all,Group.five,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all,Group.five,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all,Group.five,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.int.less
test.SMAE.int.greater

