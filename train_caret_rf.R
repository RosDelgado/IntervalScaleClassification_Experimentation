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
source("BD_OC_MAE.R")
source("BD_OC_MAEintervals.R")

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
                                               # tuneLength = 10, 
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
                                          # tuneLength = 10, 
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
  tuned.rf.caret.MAE.int.100[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf", 
                                                 # tuneLength = 10, 
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
  tuned.rf.caret.MAE.int.120[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf", 
                                                 # tuneLength = 10, 
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
Accuracy.M.MAE.int.100<-vector() 
Accuracy.M.MAE.int.120<-vector() 
#
SMAE.M.Accuracy<-vector()
SMAE.M.MAE<-vector()
SMAE.M.MAE.int.80<-vector()
SMAE.M.MAE.int.100<-vector()
SMAE.M.MAE.int.120<-vector()
#
SMAE.int.80.M.Accuracy<-vector()
SMAE.int.80.M.MAE<-vector()
SMAE.int.80.M.MAE.int.80<-vector()
#
SMAE.int.100.M.Accuracy<-vector()
SMAE.int.100.M.MAE<-vector()
SMAE.int.100.M.MAE.int.100<-vector()
#
SMAE.int.120.M.Accuracy<-vector()
SMAE.int.120.M.MAE<-vector()
SMAE.int.120.M.MAE.int.120<-vector()

for (i in 1:10)
{
  Accuracy.M.Accuracy[i]<-sum(diag(Conf.mat.tuned.rf.caret.Accuracy[[i]]))/sum(Conf.mat.tuned.rf.caret.Accuracy[[i]])
  Accuracy.M.MAE[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE[[i]])
  Accuracy.M.MAE.int.80[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.80[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.80[[i]])
  Accuracy.M.MAE.int.100[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.100[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.100[[i]])
  Accuracy.M.MAE.int.120[i]<-sum(diag(Conf.mat.tuned.rf.caret.MAE.int.120[[i]]))/sum(Conf.mat.tuned.rf.caret.MAE.int.120[[i]])
  ##
  SMAE.M.Accuracy[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.80[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.80[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.100[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.100[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.120[i]<-SMAE(mat.square(Conf.mat.tuned.rf.caret.MAE.int.120[[i]],levels.age.ordinal.encod))
  ##
  SMAE.int.80.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.80)
  SMAE.int.80.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),Len.80)
  SMAE.int.80.M.MAE.int.80[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.80[[i]],levels.age.ordinal.encod),Len.80)
  ##
  SMAE.int.100.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.100)
  SMAE.int.100.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),Len.100)
  SMAE.int.100.M.MAE.int.100[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.100[[i]],levels.age.ordinal.encod),Len.100)
  ##
  SMAE.int.120.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.120)
  SMAE.int.120.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),Len.120)
  SMAE.int.120.M.MAE.int.120[i]<-SMAE.int(mat.square(Conf.mat.tuned.rf.caret.MAE.int.120[[i]],levels.age.ordinal.encod),Len.120)
  
}

Accuracy.M.Accuracy
Accuracy.M.MAE
Accuracy.M.MAE.int.80
Accuracy.M.MAE.int.100
Accuracy.M.MAE.int.120

mean(Accuracy.M.Accuracy,na.rm=TRUE)
mean(Accuracy.M.MAE,na.rm=TRUE)
mean(Accuracy.M.MAE.int.80,na.rm=TRUE)
mean(Accuracy.M.MAE.int.100,na.rm=TRUE)
mean(Accuracy.M.MAE.int.120,na.rm=TRUE)


SMAE.M.Accuracy
SMAE.M.MAE
SMAE.M.MAE.int.80
SMAE.M.MAE.int.100
SMAE.M.MAE.int.120

mean(SMAE.M.Accuracy,na.rm=TRUE)
mean(SMAE.M.MAE,na.rm=TRUE)
mean(SMAE.M.MAE.int.80,na.rm=TRUE)
mean(SMAE.M.MAE.int.100,na.rm=TRUE)
mean(SMAE.M.MAE.int.120,na.rm=TRUE)

SMAE.int.80.M.Accuracy
SMAE.int.80.M.MAE
SMAE.int.80.M.MAE.int.80

mean(SMAE.int.80.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.80.M.MAE,na.rm=TRUE)
mean(SMAE.int.80.M.MAE.int.80,na.rm=TRUE)

SMAE.int.100.M.Accuracy
SMAE.int.100.M.MAE
SMAE.int.100.M.MAE.int.100

mean(SMAE.int.100.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.100.M.MAE,na.rm=TRUE)
mean(SMAE.int.100.M.MAE.int.100,na.rm=TRUE)

SMAE.int.120.M.Accuracy
SMAE.int.120.M.MAE
SMAE.int.120.M.MAE.int.120

mean(SMAE.int.120.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.120.M.MAE,na.rm=TRUE)
mean(SMAE.int.120.M.MAE.int.120,na.rm=TRUE)


# 
# ###############################
# ###############################
# ### ACCURACY METRIC
# 
# c1 <- rainbow(10)
# c2 <- rainbow(10, alpha=0.2)
# c3 <- rainbow(10, v=0.7)
# 
# for (i in 1:length(seeds)){
#   boxplot(Accuracy.M.Accuracy,
#           Accuracy.M.MAE,
#           Accuracy.M.MAE.int.80,
#           Accuracy.M.MAE.int.120,
#           main=paste("Accuracy metric. nnet"), 
#           xlab="metric for tuning nnet", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80", "SMAE.int.120"),
#           col=c2,medcol=c3)
#   
# }
# 
# # 
# 
# # Group.three<-c(rep(1,10),rep(2,10),rep(3,10))
# Group.four<-c(rep(1,10),rep(2,10),rep(3,10),rep(4,10))
# 
# Acc<-c(Accuracy.M.Accuracy,
#        Accuracy.M.MAE,
#        Accuracy.M.MAE.int.80,
#        Accuracy.M.MAE.int.120)
# 
# p1<-shapiro.test(Accuracy.M.Accuracy)$p.value
# p2<-shapiro.test(Accuracy.M.MAE)$p.value
# p3<-shapiro.test(Accuracy.M.MAE.int.80)$p.value
# p4<-shapiro.test(Accuracy.M.MAE.int.120)$p.value
# 
# if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05)
# {
#   test.accuracy.less<-pairwise.t.test(Acc,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
#   test.accuracy.greater<-pairwise.t.test(Acc,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
# } else {
#   test.accuracy.less<-pairwise.wilcox.test(Acc,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
#   test.accuracy.greater<-pairwise.wilcox.test(Acc,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
# }
# 
# 
# test.accuracy.less
# test.accuracy.greater
# 
# ###
# 
# 
# ###############################
# ###############################
# ### SMAE METRIC
# 
# c1 <- rainbow(10)
# c2 <- rainbow(10, alpha=0.2)
# c3 <- rainbow(10, v=0.7)
# 
# for (i in 1:length(seeds)){
#   boxplot(SMAE.M.Accuracy,
#           SMAE.M.MAE,
#           SMAE.M.MAE.int.80,
#           SMAE.M.MAE.int.120,
#           main=paste("SMAE metric. nnet"), 
#           xlab="metric for tuning nnet", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80", "SMAE.int.120"),
#           col=c2,medcol=c3)
#   
# }
# 
# # 
# 
# # Group.three<-c(rep(1,10),rep(2,10),rep(3,10))
# Group.four<-c(rep(1,10),rep(2,10),rep(3,10),rep(4,10))
# 
# 
# SMAE.all<-c(SMAE.M.Accuracy,
#             SMAE.M.MAE,
#             SMAE.M.MAE.int.80,
#             SMAE.M.MAE.int.120)
# 
# p1<-shapiro.test(SMAE.M.Accuracy)$p.value
# p2<-shapiro.test(SMAE.M.MAE)$p.value
# p3<-shapiro.test(SMAE.M.MAE.int.80)$p.value
# p4<-shapiro.test(SMAE.M.MAE.int.120)$p.value
# 
# if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05)
# {
#   test.SMAE.less<-pairwise.t.test(SMAE.all,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
#   test.SMAE.greater<-pairwise.t.test(SMAE.all,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
# } else {
#   test.SMAE.less<-pairwise.wilcox.test(SMAE.all,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
#   test.SMAE.greater<-pairwise.wilcox.test(SMAE.all,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
# }
# 
# 
# test.SMAE.less
# test.SMAE.greater
# 
# ###
# 
# ###############################
# ###############################
# ### SMAE.int.80 METRIC
# 
# c1 <- rainbow(10)
# c2 <- rainbow(10, alpha=0.2)
# c3 <- rainbow(10, v=0.7)
# 
# for (i in 1:length(seeds)){
#   boxplot(SMAE.int.80.M.Accuracy,
#           SMAE.int.80.M.MAE,
#           SMAE.int.80.M.MAE.int.80,
#           main=paste("SMAE.int metric 80. nnet"), 
#           xlab="metric for tuning nnet", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80"),
#           col=c2,medcol=c3)
#   
# }
# 
# # 
# 
# Group.three<-c(rep(1,10),rep(2,10),rep(3,10))
# 
# 
# SMAE.int.all.80<-c(SMAE.int.80.M.Accuracy,
#                    SMAE.int.80.M.MAE,
#                    SMAE.int.80.M.MAE.int.80)
# 
# p1<-shapiro.test(SMAE.int.80.M.Accuracy)$p.value
# p2<-shapiro.test(SMAE.int.80.M.MAE)$p.value
# p3<-shapiro.test(SMAE.int.80.M.MAE.int.80)$p.value
# 
# if (p1>=0.05 & p2>=0.05 & p3>=0.05)
# {
#   test.SMAE.int.less<-pairwise.t.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
#   test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
# } else {
#   test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
#   test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
# }
# 
# 
# test.SMAE.int.less
# test.SMAE.int.greater
# 
# ###############################
# ###############################
# ### SMAE.int.1200 METRIC
# 
# c1 <- rainbow(10)
# c2 <- rainbow(10, alpha=0.2)
# c3 <- rainbow(10, v=0.7)
# 
# for (i in 1:length(seeds)){
#   boxplot(SMAE.int.120.M.Accuracy,
#           SMAE.int.120.M.MAE,
#           SMAE.int.120.M.MAE.int.120,
#           main=paste("SMAE.int metric 120. nnet"), 
#           xlab="metric for tuning nnet", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.120"),
#           col=c2,medcol=c3)
#   
# }
# 
# # 
# # 
# 
# Group.three<-c(rep(1,10),rep(2,10),rep(3,10))
# 
# 
# SMAE.int.all.120<-c(SMAE.int.120.M.Accuracy,
#                     SMAE.int.120.M.MAE,
#                     SMAE.int.120.M.MAE.int.120)
# 
# p1<-shapiro.test(SMAE.int.120.M.Accuracy)$p.value
# p2<-shapiro.test(SMAE.int.120.M.MAE)$p.value
# p3<-shapiro.test(SMAE.int.120.M.MAE.int.120)$p.value
# 
# if (p1>=0.05 & p2>=0.05 & p3>=0.05)
# {
#   test.SMAE.int.less<-pairwise.t.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
#   test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
# } else {
#   test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
#   test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
# }
# 
# 
# test.SMAE.int.less
# test.SMAE.int.greater
# 


###############################
###############################
### FINAL

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.int.80.M.Accuracy,
          SMAE.int.80.M.MAE,
          SMAE.int.80.M.MAE.int.80,
          SMAE.int.100.M.MAE.int.100,
          SMAE.int.120.M.MAE.int.120,
          main=paste("SMAE.int.120"), 
          xlab="metric for tuning rf", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80", "SMAE.int.100", "SMAE.int.120"),
          col=c2,medcol=c3)
  
}

# 
# 

Group.five<-c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10))


SMAE.int.all.120<-c(SMAE.int.80.M.Accuracy,
                    SMAE.int.80.M.MAE,
                    SMAE.int.80.M.MAE.int.80,
                    SMAE.int.100.M.MAE.int.100,
                    SMAE.int.120.M.MAE.int.120)

p1<-shapiro.test(SMAE.int.80.M.Accuracy)$p.value
p2<-shapiro.test(SMAE.int.80.M.MAE)$p.value
p3<-shapiro.test(SMAE.int.80.M.MAE.int.80)$p.value
p4<-shapiro.test(SMAE.int.100.M.MAE.int.100)$p.value
p5<-shapiro.test(SMAE.int.120.M.MAE.int.120)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05 & p5>=0.05)
{
  test.SMAE.int.less<-pairwise.t.test(SMAE.int.all.120,Group.five,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all.120,Group.five,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all.120,Group.five,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all.120,Group.five,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.int.less
test.SMAE.int.greater








################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

tuned.nnet.caret.Accuracy<-list()
tuned.nnet.caret.MAE<-list()
tuned.nnet.caret.MAE.int.80<-list()
tuned.nnet.caret.MAE.int.120<-list()

nnetGrid <-  expand.grid(size = 1,
                         decay = seq(from = 0.05, to = 0.5, by = 0.05))

set.seed(12345)
random.feat<-sample(1:1024,500,replace=FALSE)

for (i in 1:10)
{ set.seed(12345)
  tuned.nnet.caret.Accuracy[[i]] <- caret::train(sub.train[[i]][ ,random.feat], 
                                                sub.train[[i]][ ,1028], 
                                                method="nnet", 
                                                # tuneLength = 10, 
                                                metric="Accuracy",
                                                tuneGrid=nnetGrid,
                                                trControl=fitControl.Accuracy)
                                             
  print(i)}


########

Conf.mat.tuned.nnet.caret.Accuracy<-list()
for (i in 1:10)
{Conf.mat.tuned.nnet.caret.Accuracy[[i]]<-
  table(predict(tuned.nnet.caret.Accuracy[[i]],sub.test[[i]][,c(random.feat)]),
        sub.test[[i]][,1028])
print(i)
}

########################

for (i in 1:10)
{ set.seed(12345)
  tuned.nnet.caret.MAE[[i]] <- caret::train(sub.train[[i]][1:1000,random.feat], 
                                                 sub.train[[i]][1:1000 ,1028], 
                                                 method="nnet", 
                                                 # tuneLength = 10, 
                                                 metric="MAE.ord",
                                                 maximize=FALSE,
                                                 tuneGrid=nnetGrid,
                                                trControl=fitControl.MAE)
  
  print(i)}

########

Conf.mat.tuned.nnet.caret.MAE<-list()
for (i in 1:10)
{Conf.mat.tuned.nnet.caret.MAE[[i]]<-
  table(predict(tuned.nnet.caret.MAE[[i]],sub.test[[i]][,c(random.feat)]),
        sub.test[[i]][,1028])
print(i)
}


########################



for (i in 1:10)
{ set.seed(12345)
  tuned.nnet.caret.MAE.int.80[[i]] <- caret::train(sub.train[[i]][1:1000,random.feat], 
                                            sub.train[[i]][1:1000 ,1028], 
                                            method="nnet", 
                                            # tuneLength = 10, 
                                            metric="MAE.int.80",
                                            maximize=FALSE,
                                            tuneGrid=nnetGrid,
                                            trControl=fitControl.MAE.int.80)
  
  print(i)}

########

Conf.mat.tuned.nnet.caret.MAE.int.80<-list()
for (i in 1:10)
{Conf.mat.tuned.nnet.caret.MAE.int.80[[i]]<-
  table(predict(tuned.nnet.caret.MAE.int.80[[i]],sub.test[[i]][,c(random.feat)]),
        sub.test[[i]][,1028])
print(i)
}


########################


for (i in 1:10)
{ set.seed(12345)
  tuned.nnet.caret.MAE.int.120[[i]] <- caret::train(sub.train[[i]][1:1000,random.feat], 
                                                   sub.train[[i]][1:1000 ,1028], 
                                                   method="nnet", 
                                                   # tuneLength = 10, 
                                                   metric="MAE.int.120",
                                                   maximize=FALSE,
                                                   tuneGrid=nnetGrid,
                                                   trControl=fitControl.MAE.int.120)
  
  print(i)}

########

Conf.mat.tuned.nnet.caret.MAE.int.120<-list()
for (i in 1:10)
{Conf.mat.tuned.nnet.caret.MAE.int.120[[i]]<-
  table(predict(tuned.nnet.caret.MAE.int.120[[i]],sub.test[[i]][,c(random.feat)]),
        sub.test[[i]][,1028])
print(i)
}


########################

################################################################################
################################################################################
################## Confusion matrices, Accuracy, SMAE and SMAE.int
################################################################################
#
Accuracy.M.Accuracy<-vector()
Accuracy.M.MAE<-vector()
Accuracy.M.MAE.int.80<-vector() 
Accuracy.M.MAE.int.120<-vector() 
#
SMAE.M.Accuracy<-vector()
SMAE.M.MAE<-vector()
SMAE.M.MAE.int.80<-vector()
SMAE.M.MAE.int.120<-vector()
#
SMAE.int.80.M.Accuracy<-vector()
SMAE.int.80.M.MAE<-vector()
SMAE.int.80.M.MAE.int.80<-vector()
#
SMAE.int.120.M.Accuracy<-vector()
SMAE.int.120.M.MAE<-vector()
SMAE.int.120.M.MAE.int.120<-vector()

for (i in 1:10)
{
  Accuracy.M.Accuracy[i]<-sum(diag(Conf.mat.tuned.nnet.caret.Accuracy[[i]]))/sum(Conf.mat.tuned.nnet.caret.Accuracy[[i]])
  Accuracy.M.MAE[i]<-sum(diag(Conf.mat.tuned.nnet.caret.MAE[[i]]))/sum(Conf.mat.tuned.nnet.caret.MAE[[i]])
  Accuracy.M.MAE.int.80[i]<-sum(diag(Conf.mat.tuned.nnet.caret.MAE.int.80[[i]]))/sum(Conf.mat.tuned.nnet.caret.MAE.int.80[[i]])
  Accuracy.M.MAE.int.120[i]<-sum(diag(Conf.mat.tuned.nnet.caret.MAE.int.120[[i]]))/sum(Conf.mat.tuned.nnet.caret.MAE.int.120[[i]])
  ##
  SMAE.M.Accuracy[i]<-SMAE(mat.square(Conf.mat.tuned.nnet.caret.Accuracy[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE[i]<-SMAE(mat.square(Conf.mat.tuned.nnet.caret.MAE[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.80[i]<-SMAE(mat.square(Conf.mat.tuned.nnet.caret.MAE.int.80[[i]],levels.age.ordinal.encod))
  SMAE.M.MAE.int.120[i]<-SMAE(mat.square(Conf.mat.tuned.nnet.caret.MAE.int.120[[i]],levels.age.ordinal.encod))
  ##
  SMAE.int.80.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.nnet.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.80)
  SMAE.int.80.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.nnet.caret.MAE[[i]],levels.age.ordinal.encod),Len.80)
  SMAE.int.80.M.MAE.int.80[i]<-SMAE.int(mat.square(Conf.mat.tuned.nnet.caret.MAE.int.80[[i]],levels.age.ordinal.encod),Len.80)
  ##
  SMAE.int.120.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.nnet.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.120)
  SMAE.int.120.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.nnet.caret.MAE[[i]],levels.age.ordinal.encod),Len.120)
  SMAE.int.120.M.MAE.int.120[i]<-SMAE.int(mat.square(Conf.mat.tuned.nnet.caret.MAE.int.120[[i]],levels.age.ordinal.encod),Len.120)
  
}

Accuracy.M.Accuracy
Accuracy.M.MAE
Accuracy.M.MAE.int.80
Accuracy.M.MAE.int.120

mean(Accuracy.M.Accuracy,na.rm=TRUE)
mean(Accuracy.M.MAE,na.rm=TRUE)
mean(Accuracy.M.MAE.int.80,na.rm=TRUE)
mean(Accuracy.M.MAE.int.120,na.rm=TRUE)


SMAE.M.Accuracy
SMAE.M.MAE
SMAE.M.MAE.int.80
SMAE.M.MAE.int.120

mean(SMAE.M.Accuracy,na.rm=TRUE)
mean(SMAE.M.MAE,na.rm=TRUE)
mean(SMAE.M.MAE.int.80,na.rm=TRUE)
mean(SMAE.M.MAE.int.120,na.rm=TRUE)

SMAE.int.80.M.Accuracy
SMAE.int.80.M.MAE
SMAE.int.80.M.MAE.int.80

mean(SMAE.int.80.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.80.M.MAE,na.rm=TRUE)
mean(SMAE.int.80.M.MAE.int.80,na.rm=TRUE)

SMAE.int.120.M.Accuracy
SMAE.int.120.M.MAE
SMAE.int.120.M.MAE.int.120

mean(SMAE.int.120.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.120.M.MAE,na.rm=TRUE)
mean(SMAE.int.120.M.MAE.int.120,na.rm=TRUE)



###############################
###############################
### ACCURACY METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(Accuracy.M.Accuracy,
          Accuracy.M.MAE,
          Accuracy.M.MAE.int.80,
          Accuracy.M.MAE.int.120,
          main=paste("Accuracy metric. nnet"), 
          xlab="metric for tuning nnet", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80", "SMAE.int.120"),
          col=c2,medcol=c3)
  
}

# 

# Group.three<-c(rep(1,10),rep(2,10),rep(3,10))
Group.four<-c(rep(1,10),rep(2,10),rep(3,10),rep(4,10))

Acc<-c(Accuracy.M.Accuracy,
       Accuracy.M.MAE,
       Accuracy.M.MAE.int.80,
       Accuracy.M.MAE.int.120)

p1<-shapiro.test(Accuracy.M.Accuracy)$p.value
p2<-shapiro.test(Accuracy.M.MAE)$p.value
p3<-shapiro.test(Accuracy.M.MAE.int.80)$p.value
p4<-shapiro.test(Accuracy.M.MAE.int.120)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05)
{
  test.accuracy.less<-pairwise.t.test(Acc,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.accuracy.greater<-pairwise.t.test(Acc,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.accuracy.less<-pairwise.wilcox.test(Acc,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.accuracy.greater<-pairwise.wilcox.test(Acc,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.accuracy.less
test.accuracy.greater

###


###############################
###############################
### SMAE METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.M.Accuracy,
          SMAE.M.MAE,
          SMAE.M.MAE.int.80,
          SMAE.M.MAE.int.120,
          main=paste("SMAE metric. nnet"), 
          xlab="metric for tuning nnet", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80", "SMAE.int.120"),
          col=c2,medcol=c3)
  
}

# 

# Group.three<-c(rep(1,10),rep(2,10),rep(3,10))
Group.four<-c(rep(1,10),rep(2,10),rep(3,10),rep(4,10))


SMAE.all<-c(SMAE.M.Accuracy,
            SMAE.M.MAE,
            SMAE.M.MAE.int.80,
            SMAE.M.MAE.int.120)

p1<-shapiro.test(SMAE.M.Accuracy)$p.value
p2<-shapiro.test(SMAE.M.MAE)$p.value
p3<-shapiro.test(SMAE.M.MAE.int.80)$p.value
p4<-shapiro.test(SMAE.M.MAE.int.120)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05)
{
  test.SMAE.less<-pairwise.t.test(SMAE.all,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.greater<-pairwise.t.test(SMAE.all,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.less<-pairwise.wilcox.test(SMAE.all,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.greater<-pairwise.wilcox.test(SMAE.all,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.less
test.SMAE.greater

###

###############################
###############################
### SMAE.int.80 METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.int.80.M.Accuracy,
          SMAE.int.80.M.MAE,
          SMAE.int.80.M.MAE.int.80,
          main=paste("SMAE.int metric 80. nnet"), 
          xlab="metric for tuning nnet", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80"),
          col=c2,medcol=c3)
  
}

# 

Group.three<-c(rep(1,10),rep(2,10),rep(3,10))


SMAE.int.all.80<-c(SMAE.int.80.M.Accuracy,
                   SMAE.int.80.M.MAE,
                   SMAE.int.80.M.MAE.int.80)

p1<-shapiro.test(SMAE.int.80.M.Accuracy)$p.value
p2<-shapiro.test(SMAE.int.80.M.MAE)$p.value
p3<-shapiro.test(SMAE.int.80.M.MAE.int.80)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05)
{
  test.SMAE.int.less<-pairwise.t.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.int.less
test.SMAE.int.greater

###############################
###############################
### SMAE.int.1200 METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.int.120.M.Accuracy,
          SMAE.int.120.M.MAE,
          SMAE.int.120.M.MAE.int.120,
          main=paste("SMAE.int metric 120. nnet"), 
          xlab="metric for tuning nnet", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.120"),
          col=c2,medcol=c3)
  
}

# 
# 

Group.three<-c(rep(1,10),rep(2,10),rep(3,10))


SMAE.int.all.120<-c(SMAE.int.120.M.Accuracy,
                    SMAE.int.120.M.MAE,
                    SMAE.int.120.M.MAE.int.120)

p1<-shapiro.test(SMAE.int.120.M.Accuracy)$p.value
p2<-shapiro.test(SMAE.int.120.M.MAE)$p.value
p3<-shapiro.test(SMAE.int.120.M.MAE.int.120)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05)
{
  test.SMAE.int.less<-pairwise.t.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.int.less
test.SMAE.int.greater



















for (i in 1:5)
{ set.seed(12345)
  tuned.knn.caret.Accuracy[[i]] <- caret::train(sub.train[[i]][,-c(1025:1028)], 
                                           sub.train[[i]][,1028], 
                                           method="knn", 
                                           tuneLength = 10, 
                                           metric="Accuracy")
print(i)}


########

Conf.mat.tuned.knn.caret.Accuracy<-list()
for (i in 1:5)
{Conf.mat.tuned.knn.caret.Accuracy[[i]]<-
table(class::knn(train=sub.train[[i]][,-c(1025:1027)], test=test[[i]][,-c(1025:1027)],cl=sub.train[[i]][,1028],
                         k=tuned.knn.caret.Accuracy[[i]]$bestTune,l=round(tuned.knn.caret.Accuracy[[i]]$bestTune/2),
                         prob=FALSE, use.all=TRUE),test=test[[i]][,1028])
print(i)
}
              

#####
##

for (i in 1:5)
{
  set.seed(12345)
  tuned.knn.caret.MAE[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                               sub.train[[i]][ ,1028], 
                                               method="knn", 
                                               tuneLength = 10, 
                                               metric="MAE.ord",
                                               maximize=FALSE,
                                               trControl=fitControl.MAE)
print(i)}

 
########
Conf.mat.tuned.knn.caret.MAE<-list()
for (i in 1:5)
{Conf.mat.tuned.knn.caret.MAE[[i]]<-
  table(class::knn(train=sub.train[[i]][,-c(1025:1027)], test=test[[i]][,-c(1025:1027)],cl=sub.train[[i]][,1028],
                   k=tuned.knn.caret.MAE[[i]]$bestTune,l=round(tuned.knn.caret.MAE[[i]]$bestTune/2),
                   prob=FALSE, use.all=TRUE),test=test[[i]][,1028])
print(i)
}

##

for (i in 1:5)
{
  set.seed(12345)
  tuned.knn.caret.MAE.int.80[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                           sub.train[[i]][ ,1028], 
                                           method="knn", 
                                           tuneLength = 10, 
                                           metric="MAE.int.80",
                                           maximize=FALSE,
                                           trControl=fitControl.MAE.int.80)
  print(i)}

#### 


Conf.mat.tuned.knn.caret.MAE.int.80<-list()
for (i in 1:5)
{Conf.mat.tuned.knn.caret.MAE.int.80[[i]]<-
  table(class::knn(train=sub.train[[i]][,-c(1025:1027)], test=test[[i]][,-c(1025:1027)],cl=sub.train[[i]][,1028],
                   k=tuned.knn.caret.MAE.int.80[[i]]$bestTune,l=round(tuned.knn.caret.MAE.int.80[[i]]$bestTune/2),
                   prob=FALSE, use.all=TRUE),test=test[[i]][,1028])
print(i)
}

##

##

for (i in 1:5)
{
  set.seed(12345)
  tuned.knn.caret.MAE.int.120[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                  sub.train[[i]][ ,1028], 
                                                  method="knn", 
                                                  tuneLength = 10, 
                                                  metric="MAE.int.120",
                                                  maximize=FALSE,
                                                  trControl=fitControl.MAE.int.120)
  print(i)}

#### 

Conf.mat.tuned.knn.caret.MAE.int.120<-list()
for (i in 1:5)
{Conf.mat.tuned.knn.caret.MAE.int.120[[i]]<-
  table(class::knn(train=sub.train[[i]][,-c(1025:1027)], test=test[[i]][,-c(1025:1027)],cl=sub.train[[i]][,1028],
                   k=tuned.knn.caret.MAE.int.120[[i]]$bestTune,l=round(tuned.knn.caret.MAE.int.120[[i]]$bestTune/2),
                   prob=FALSE, use.all=TRUE),test=test[[i]][,1028])
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
Accuracy.M.MAE.int.120<-vector() 
#
SMAE.M.Accuracy<-vector()
SMAE.M.MAE<-vector()
SMAE.M.MAE.int.80<-vector()
SMAE.M.MAE.int.120<-vector()
#
SMAE.int.80.M.Accuracy<-vector()
SMAE.int.80.M.MAE<-vector()
SMAE.int.80.M.MAE.int.80<-vector()
#
SMAE.int.120.M.Accuracy<-vector()
SMAE.int.120.M.MAE<-vector()
SMAE.int.120.M.MAE.int.120<-vector()

for (i in 1:5)
{
Accuracy.M.Accuracy[i]<-sum(diag(Conf.mat.tuned.knn.caret.Accuracy[[i]]))/sum(Conf.mat.tuned.knn.caret.Accuracy[[i]])
Accuracy.M.MAE[i]<-sum(diag(Conf.mat.tuned.knn.caret.MAE[[i]]))/sum(Conf.mat.tuned.knn.caret.MAE[[i]])
Accuracy.M.MAE.int.80[i]<-sum(diag(Conf.mat.tuned.knn.caret.MAE.int.80[[i]]))/sum(Conf.mat.tuned.knn.caret.MAE.int.80[[i]])
Accuracy.M.MAE.int.120[i]<-sum(diag(Conf.mat.tuned.knn.caret.MAE.int.120[[i]]))/sum(Conf.mat.tuned.knn.caret.MAE.int.120[[i]])
##
SMAE.M.Accuracy[i]<-SMAE(mat.square(Conf.mat.tuned.knn.caret.Accuracy[[i]],levels.age.ordinal.encod))
SMAE.M.MAE[i]<-SMAE(mat.square(Conf.mat.tuned.knn.caret.MAE[[i]],levels.age.ordinal.encod))
SMAE.M.MAE.int.80[i]<-SMAE(mat.square(Conf.mat.tuned.knn.caret.MAE.int.80[[i]],levels.age.ordinal.encod))
SMAE.M.MAE.int.120[i]<-SMAE(mat.square(Conf.mat.tuned.knn.caret.MAE.int.120[[i]],levels.age.ordinal.encod))
##
SMAE.int.80.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.80)
SMAE.int.80.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.caret.MAE[[i]],levels.age.ordinal.encod),Len.80)
SMAE.int.80.M.MAE.int.80[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.caret.MAE.int.80[[i]],levels.age.ordinal.encod),Len.80)
##
SMAE.int.120.M.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.caret.Accuracy[[i]],levels.age.ordinal.encod),Len.120)
SMAE.int.120.M.MAE[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.caret.MAE[[i]],levels.age.ordinal.encod),Len.120)
SMAE.int.120.M.MAE.int.120[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.caret.MAE.int.120[[i]],levels.age.ordinal.encod),Len.120)

}

Accuracy.M.Accuracy
Accuracy.M.MAE
Accuracy.M.MAE.int.80
Accuracy.M.MAE.int.120

mean(Accuracy.M.Accuracy,na.rm=TRUE)
mean(Accuracy.M.MAE,na.rm=TRUE)
mean(Accuracy.M.MAE.int.80,na.rm=TRUE)
mean(Accuracy.M.MAE.int.120,na.rm=TRUE)


SMAE.M.Accuracy
SMAE.M.MAE
SMAE.M.MAE.int.80
SMAE.M.MAE.int.120

mean(SMAE.M.Accuracy,na.rm=TRUE)
mean(SMAE.M.MAE,na.rm=TRUE)
mean(SMAE.M.MAE.int.80,na.rm=TRUE)
mean(SMAE.M.MAE.int.120,na.rm=TRUE)

SMAE.int.80.M.Accuracy
SMAE.int.80.M.MAE
SMAE.int.80.M.MAE.int.80

mean(SMAE.int.80.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.80.M.MAE,na.rm=TRUE)
mean(SMAE.int.80.M.MAE.int.80,na.rm=TRUE)

SMAE.int.120.M.Accuracy
SMAE.int.120.M.MAE
SMAE.int.120.M.MAE.int.120

mean(SMAE.int.120.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.120.M.MAE,na.rm=TRUE)
mean(SMAE.int.120.M.MAE.int.120,na.rm=TRUE)



###############################
###############################
### ACCURACY METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(Accuracy.M.Accuracy,
          Accuracy.M.MAE,
          Accuracy.M.MAE.int.80,
          Accuracy.M.MAE.int.120,
          main=paste("Accuracy metric. knn"), 
          xlab="metric for tuning knn", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80", "SMAE.int.120"),
          col=c2,medcol=c3)
  
}

# 

# Group.three<-c(rep(1,10),rep(2,10),rep(3,10))
Group.four<-c(rep(1,5),rep(2,5),rep(3,5),rep(4,5))

Acc<-c(Accuracy.M.Accuracy,
       Accuracy.M.MAE,
       Accuracy.M.MAE.int.80,
       Accuracy.M.MAE.int.120)

p1<-shapiro.test(Accuracy.M.Accuracy)$p.value
p2<-shapiro.test(Accuracy.M.MAE)$p.value
p3<-shapiro.test(Accuracy.M.MAE.int.80)$p.value
p4<-shapiro.test(Accuracy.M.MAE.int.120)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05)
{
  test.accuracy.less<-pairwise.t.test(Acc,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.accuracy.greater<-pairwise.t.test(Acc,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.accuracy.less<-pairwise.wilcox.test(Acc,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.accuracy.greater<-pairwise.wilcox.test(Acc,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.accuracy.less
test.accuracy.greater

###


###############################
###############################
### SMAE METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.M.Accuracy,
          SMAE.M.MAE,
          SMAE.M.MAE.int.80,
          SMAE.M.MAE.int.120,
          main=paste("SMAE metric. knn"), 
          xlab="metric for tuning knn", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80", "SMAE.int.120"),
          col=c2,medcol=c3)
  
}

# 

# Group.three<-c(rep(1,10),rep(2,10),rep(3,10))
Group.four<-c(rep(1,5),rep(2,5),rep(3,5),rep(4,5))


SMAE.all<-c(SMAE.M.Accuracy,
            SMAE.M.MAE,
            SMAE.M.MAE.int.80,
            SMAE.M.MAE.int.120)

p1<-shapiro.test(SMAE.M.Accuracy)$p.value
p2<-shapiro.test(SMAE.M.MAE)$p.value
p3<-shapiro.test(SMAE.M.MAE.int.80)$p.value
p4<-shapiro.test(SMAE.M.MAE.int.120)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05)
{
  test.SMAE.less<-pairwise.t.test(SMAE.all,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.greater<-pairwise.t.test(SMAE.all,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.less<-pairwise.wilcox.test(SMAE.all,Group.four,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.greater<-pairwise.wilcox.test(SMAE.all,Group.four,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.less
test.SMAE.greater

###

###############################
###############################
### SMAE.int.80 METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.int.80.M.Accuracy,
          SMAE.int.80.M.MAE,
          SMAE.int.80.M.MAE.int.80,
          main=paste("SMAE.int metric 80. knn"), 
          xlab="metric for tuning knn", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.80"),
          col=c2,medcol=c3)
  
}

# 

Group.three<-c(rep(1,5),rep(2,5),rep(3,5))


SMAE.int.all.80<-c(SMAE.int.80.M.Accuracy,
                SMAE.int.80.M.MAE,
                SMAE.int.80.M.MAE.int.80)

p1<-shapiro.test(SMAE.int.80.M.Accuracy)$p.value
p2<-shapiro.test(SMAE.int.80.M.MAE)$p.value
p3<-shapiro.test(SMAE.int.80.M.MAE.int.80)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05)
{
  test.SMAE.int.less<-pairwise.t.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all.80,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.int.less
test.SMAE.int.greater

###############################
###############################
### SMAE.int.1200 METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.int.120.M.Accuracy,
          SMAE.int.120.M.MAE,
          SMAE.int.120.M.MAE.int.120,
          main=paste("SMAE.int metric 120. knn"), 
          xlab="metric for tuning knn", ylab=" ",names=c("Accuracy", "SMAE", "SMAE.int.120"),
          col=c2,medcol=c3)
  
}

# 
# 

Group.three<-c(rep(1,5),rep(2,5),rep(3,5))


SMAE.int.all.120<-c(SMAE.int.120.M.Accuracy,
                SMAE.int.120.M.MAE,
                SMAE.int.120.M.MAE.int.120)

p1<-shapiro.test(SMAE.int.120.M.Accuracy)$p.value
p2<-shapiro.test(SMAE.int.120.M.MAE)$p.value
p3<-shapiro.test(SMAE.int.120.M.MAE.int.120)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05)
{
  test.SMAE.int.less<-pairwise.t.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all.120,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.int.less
test.SMAE.int.greater















################################################################################
#############   Save Accuracy, SMAE and SMAE.int values
################################################################################

Accuracy.M.Accuracy.knn.l.13<-Accuracy.M.Accuracy
Accuracy.M.MAE.knn.l.13<-Accuracy.M.MAE
Accuracy.M.MAE.int.knn.l.13<-Accuracy.M.MAE.int

SMAE.M.Accuracy.knn.l.13<-SMAE.M.Accuracy
SMAE.M.MAE.knn.l.13<-SMAE.M.MAE
SMAE.M.MAE.int.knn.l.13<-SMAE.M.MAE.int

SMAE.int.M.Accuracy.knn.l.13<-SMAE.int.M.Accuracy
SMAE.int.M.MAE.knn.l.13<-SMAE.int.M.MAE
SMAE.int.M.MAE.int.knn.l.13<-SMAE.int.M.MAE.int







# 
# tuned.svm.MSE <- e1071::tune(svm,age.bin.num ~., data = df[random.sampl,c(feat,1027)], 
#             ranges = list(gamma = 2^(-2:2), cost = 2^(2:5)),
#             tunecontrol = tune.control(sampling = "fix",error.fun=NULL)
# )
# 
# 
# tuned.svm.MAE.int.110<-tuned.svm.MAE.int
# 
# summary(tuned.svm.MSE)
# plot(tuned.svm.MSE)
# tuned.svm.MSE$best.parameters
# tuned.svm.MSE$performances
# tuned.svm.MSE$best.performance
# tuned.svm.MSE$method
# tuned.svm.MSE$nparcomb
# tuned.svm.MSE$sampling
# tuned.svm.MSE$best.model
# 
# 
# best.svm.MAE.int<-tuned.svm.MAE.int$best.model
# best.svm.MAE<-tuned.svm.MAE$best.model
# best.svm.Accuracy<-tuned.svm.Accuracy$best.model
# 

best.svm.Accuracy<-tuned.svm.Accuracy$best.model

table(predict(best.svm.Accuracy, df[test.set.rows[[i]],-c(1025:1027)], decision.values = FALSE,
              probability = FALSE, na.action = na.omit),df[test.set.rows[[i]],1028])
#



best.svm.MAE<-tuned.svm.MAE$best.model

table(predict(best.svm.MAE, df[test.set.rows[[i]],-c(1025:1027)], decision.values = FALSE,
              probability = FALSE, na.action = na.omit),df[test.set.rows[[i]],1028])
#



best.svm.MAE.int.120<-tuned.svm.MAE.int.120$best.model

table(predict(best.svm.MAE.int.120, df[test.set.rows[[i]],-c(1025:1027)], decision.values = FALSE,
        probability = FALSE, na.action = na.omit),df[test.set.rows[[i]],1028])
#
# 
best.svm.MAE.int.80<-tuned.svm.MAE.int.80$best.model

table(predict(best.svm.MAE.int.80, df[test.set.rows[[i]],-c(1025:1027)], decision.values = FALSE,
        probability = FALSE, na.action = na.omit),df[test.set.rows[[i]],1028])



# 
# #################


################################################################################
################################################################################
########## e1071::tune.rf. Resampling method: cross-validation
################################################################################

tuned.rf.Accuracy <- e1071::tune(randomForest,age.bin.num.factor~., data= df[random.sampl ,-c(1025:1027)],  
                                  ntree=1:10,
                                  tunecontrol = tune.control(sampling = "cross", cross=5, error.fun=NULL))  # error.fun = Error rate

tuned.svm.MAE <- e1071::tune(svm, age.bin.num.factor~., data= df[random.sampl ,-c(1025:1027)],  
                             ranges=list(gamma = 2^(-4:2), cost = 2^(1:3)),
                             tunecontrol = tune.control(sampling = "cross", cross=5, error.fun=standard.MAE.ord))  # error.fun = MAE

tuned.svm.MAE.int.120 <- e1071::tune(svm, age.bin.num.factor~., data= df[random.sampl ,-c(1025:1027)],  
                                     ranges=list(gamma = 2^(-4:2), cost = 2^(1:3)),
                                     tunecontrol = tune.control(sampling = "cross", cross=5, error.fun=standard.MAE.int))  # error.fun = MAE.int


tuned.svm.MAE.int.80 <- e1071::tune(svm, age.bin.num.factor~., data= df[random.sampl ,-c(1025:1027)],  
                                    ranges=list(gamma = 2^(-4:2), cost = 2^(1:3)),
                                    tunecontrol = tune.control(sampling = "cross", cross=5, error.fun=standard.MAE.int))  # error.fun = MAE.int





# 
# tuned.svm.MSE <- e1071::tune(svm,age.bin.num ~., data = df[random.sampl,c(feat,1027)], 
#             ranges = list(gamma = 2^(-2:2), cost = 2^(2:5)),
#             tunecontrol = tune.control(sampling = "fix",error.fun=NULL)
# )
# 
# 
# tuned.svm.MAE.int.110<-tuned.svm.MAE.int
# 
# summary(tuned.svm.MSE)
# plot(tuned.svm.MSE)
# tuned.svm.MSE$best.parameters
# tuned.svm.MSE$performances
# tuned.svm.MSE$best.performance
# tuned.svm.MSE$method
# tuned.svm.MSE$nparcomb
# tuned.svm.MSE$sampling
# tuned.svm.MSE$best.model
# 
# 
# best.svm.MAE.int<-tuned.svm.MAE.int$best.model
# best.svm.MAE<-tuned.svm.MAE$best.model
# best.svm.Accuracy<-tuned.svm.Accuracy$best.model
# 

best.svm.Accuracy<-tuned.svm.Accuracy$best.model

table(predict(best.svm.Accuracy, df[test.set.rows[[i]],-c(1025:1027)], decision.values = FALSE,
              probability = FALSE, na.action = na.omit),df[test.set.rows[[i]],1028])
#



best.svm.MAE<-tuned.svm.MAE$best.model

table(predict(best.svm.MAE, df[test.set.rows[[i]],-c(1025:1027)], decision.values = FALSE,
              probability = FALSE, na.action = na.omit),df[test.set.rows[[i]],1028])
#



best.svm.MAE.int.120<-tuned.svm.MAE.int.120$best.model

table(predict(best.svm.MAE.int.120, df[test.set.rows[[i]],-c(1025:1027)], decision.values = FALSE,
              probability = FALSE, na.action = na.omit),df[test.set.rows[[i]],1028])
#
# 
best.svm.MAE.int.80<-tuned.svm.MAE.int.80$best.model

table(predict(best.svm.MAE.int.80, df[test.set.rows[[i]],-c(1025:1027)], decision.values = FALSE,
              probability = FALSE, na.action = na.omit),df[test.set.rows[[i]],1028])



# 
# #################













set.seed(12345)
random.sampl<-sample(1:(dim(df)[1]),2000,replace=FALSE)
# random.sampl.2 <-sample(setdiff(1:dim(df)[1],random.sampl.1),2000,replace=FALSE)
# random.sampl.3 <-sample(setdiff(1:dim(df)[1],c(random.sampl.1,random.sampl.2)),2000,replace=FALSE)
# random.sampl<-setdiff(1:dim(df)[1],c(random.sampl.1,random.sampl.2,random.sampl.3))

################################################################################
################################################################################
########## e1071::tune.knn. Resampling method: cross-validation
################################################################################

tuned.knn.Accuracy <- e1071::tune.knn(x = df[random.sampl ,-c(1025:1028)], y = df[random.sampl ,1028], 
                                      k=1:20,
                                  tunecontrol = tune.control(sampling = "cross", cross=5, error.fun=NULL))  # error.fun = Error rate


tuned.knn.MAE <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                   k=1:20,
                                   tunecontrol = tune.control(sampling = "cross", cross=5, error.fun=standard.MAE.ord))
 

tuned.knn.MAE.int <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                       k=1:20, 
                                       tunecontrol = tune.control(sampling = "cross",cross=5, error.fun=standard.MAE.int))
 




summary(tuned.knn.Accuracy)
plot(tuned.knn.Accuracy)
tuned.knn.Accuracy$best.parameters

summary(tuned.knn.MAE)
plot(tuned.knn.MAE)
tuned.knn.MAE$best.parameters

summary(tuned.knn.MAE.int)
plot(tuned.knn.MAE.int)
tuned.knn.MAE.int$best.parameters



################################################################################
################################################################################
########## predictions (validation) with class::knn
################################################################################

l.max<-min(tuned.knn.Accuracy$best.parameters,tuned.knn.MAE$best.parameters,tuned.knn.MAE.int$best.parameters)

r=0:l.max

h=8


library(class)

set.seed(12345)
seeds<-sample(1:10000,10,replace=FALSE)

pred.test.knn.Accuracy<-list()
pred.test.knn.MAE<-list()
pred.test.knn.MAE.int<-list()
test.set.rows<-list()

for (i in 1:length(seeds))
{
  yyy <- setdiff(1:nrow(df), random.sampl)
  set.seed(seeds[i])
  test.set.rows[[i]]<-sample(yyy,500,replace=FALSE)

 pred.test.knn.Accuracy[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
           k=tuned.knn.Accuracy$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
 
 pred.test.knn.MAE[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
                                         k=tuned.knn.MAE$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
 
 pred.test.knn.MAE.int[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
                                         k=tuned.knn.MAE.int$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
 
 print(i)
}





################################################################################
################################################################################
################## Confusion matrices, Accuracy, SMAE and SMAE.int
################################################################################

M.Accuracy<-list()
M.MAE<-list()
M.MAE.int<-list()
#
Accuracy.M.Accuracy<-vector()
Accuracy.M.MAE<-vector()
Accuracy.M.MAE<-vector() 
#
SMAE.M.Accuracy<-vector()
SMAE.M.MAE<-vector()
SMAE.M.MAE.int<-vector()
#
SMAE.int.M.Accuracy<-vector()
SMAE.int.M.MAE<-vector()
SMAE.int.M.MAE.int<-vector()

for (i in 1:length(seeds))
{M.Accuracy[[i]]<-table(pred.test.knn.Accuracy[[i]],df[test.set.rows[[i]],1028])
M.MAE[[i]]<-table(pred.test.knn.MAE[[i]],df[test.set.rows[[i]],1028])
M.MAE.int[[i]]<-table(pred.test.knn.MAE.int[[i]],df[test.set.rows[[i]],1028])
##
Accuracy.M.Accuracy[i]<-sum(diag(M.Accuracy[[i]]))/sum(M.Accuracy[[i]])
Accuracy.M.MAE[i]<-sum(diag(M.MAE[[i]]))/sum(M.MAE[[i]])
Accuracy.M.MAE.int[i]<-sum(diag(M.MAE.int[[i]]))/sum(M.MAE.int[[i]])
##
SMAE.M.Accuracy[i]<-SMAE(mat.square(M.Accuracy[[i]],levels.age.ordinal.encod))
SMAE.M.MAE[i]<-SMAE(mat.square(M.MAE[[i]],levels.age.ordinal.encod))
SMAE.M.MAE.int[i]<-SMAE(mat.square(M.MAE.int[[i]],levels.age.ordinal.encod))
##
SMAE.int.M.Accuracy[i]<-SMAE.int(mat.square(M.Accuracy[[i]],levels.age.ordinal.encod),Len)
SMAE.int.M.MAE[i]<-SMAE.int(mat.square(M.MAE[[i]],levels.age.ordinal.encod),Len)
SMAE.int.M.MAE.int[i]<-SMAE.int(mat.square(M.MAE.int[[i]],levels.age.ordinal.encod),Len)
}

Accuracy.M.Accuracy
Accuracy.M.MAE
Accuracy.M.MAE.int

mean(Accuracy.M.Accuracy,na.rm=TRUE)
mean(Accuracy.M.MAE,na.rm=TRUE)
mean(Accuracy.M.MAE.int,na.rm=TRUE)



SMAE.M.Accuracy
SMAE.M.MAE
SMAE.M.MAE.int

mean(SMAE.M.Accuracy,na.rm=TRUE)
mean(SMAE.M.MAE,na.rm=TRUE)
mean(SMAE.M.MAE.int,na.rm=TRUE)


SMAE.int.M.Accuracy
SMAE.int.M.MAE
SMAE.int.M.MAE.int

mean(SMAE.int.M.Accuracy,na.rm=TRUE)
mean(SMAE.int.M.MAE,na.rm=TRUE)
mean(SMAE.int.M.MAE.int,na.rm=TRUE)


################################################################################
#############   Save Accuracy, SMAE and SMAE.int values
################################################################################

Accuracy.M.Accuracy.knn.l.13<-Accuracy.M.Accuracy
Accuracy.M.MAE.knn.l.13<-Accuracy.M.MAE
Accuracy.M.MAE.int.knn.l.13<-Accuracy.M.MAE.int

SMAE.M.Accuracy.knn.l.13<-SMAE.M.Accuracy
SMAE.M.MAE.knn.l.13<-SMAE.M.MAE
SMAE.M.MAE.int.knn.l.13<-SMAE.M.MAE.int

SMAE.int.M.Accuracy.knn.l.13<-SMAE.int.M.Accuracy
SMAE.int.M.MAE.knn.l.13<-SMAE.int.M.MAE
SMAE.int.M.MAE.int.knn.l.13<-SMAE.int.M.MAE.int


###############################
###############################
### ACCURACY METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
    boxplot(Accuracy.M.Accuracy,
            Accuracy.M.MAE,
            Accuracy.M.MAE.int,
            main=paste("Accuracy metric. knn with l=13"), 
            xlab="error.fun for tuning knn", ylab=" ",names=c("Error.rate", "SMAE", "SMAE.int"),
            col=c2,medcol=c3)
    
  }

# 

Group.three<-c(rep(1,10),rep(2,10),rep(3,10))

    Acc<-c(Accuracy.M.Accuracy,
          Accuracy.M.MAE,
          Accuracy.M.MAE.int)
    
    p1<-shapiro.test(Accuracy.M.Accuracy)$p.value
    p2<-shapiro.test(Accuracy.M.MAE)$p.value
    p3<-shapiro.test(Accuracy.M.MAE.int)$p.value
    
    if (p1>=0.05 & p2>=0.05 & p3>=0.05)
    {
      test.accuracy.less<-pairwise.t.test(Acc,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
      test.accuracy.greater<-pairwise.t.test(Acc,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
    } else {
      test.accuracy.less<-pairwise.wilcox.test(Acc,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
      test.accuracy.greater<-pairwise.wilcox.test(Acc,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
    }
    
    
    test.accuracy.less
    test.accuracy.greater

    ###


###############################
###############################
### SMAE METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.M.Accuracy,
          SMAE.M.MAE,
          SMAE.M.MAE.int,
          main=paste("SMAE metric. knn with l=13"), 
          xlab="error.fun for tuning knn", ylab=" ",names=c("Error.rate", "SMAE", "SMAE.int"),
          col=c2,medcol=c3)
  
}

# 

Group.three<-c(rep(1,10),rep(2,10),rep(3,10))


  SMAE.all<-c(SMAE.M.Accuracy,
         SMAE.M.MAE,
         SMAE.M.MAE.int)
  
  p1<-shapiro.test(SMAE.M.Accuracy)$p.value
  p2<-shapiro.test(SMAE.M.MAE)$p.value
  p3<-shapiro.test(SMAE.M.MAE.int)$p.value
  
  if (p1>=0.05 & p2>=0.05 & p3>=0.05)
  {
    test.SMAE.less<-pairwise.t.test(SMAE.all,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
    test.SMAE.greater<-pairwise.t.test(SMAE.all,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
  } else {
    test.SMAE.less<-pairwise.wilcox.test(SMAE.all,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
    test.SMAE.greater<-pairwise.wilcox.test(SMAE.all,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
  }
  
  
  test.SMAE.less
  test.SMAE.greater
 
###

###############################
###############################
### SMAE.int METRIC

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.int.M.Accuracy,
          SMAE.int.M.MAE,
          SMAE.int.M.MAE.int,
          main=paste("SMAE.int metric. knn with l=13"), 
          xlab="error.fun for tuning knn", ylab=" ",names=c("Error.rate", "SMAE", "SMAE.int"),
          col=c2,medcol=c3)
  
}

# 

Group.three<-c(rep(1,10),rep(2,10),rep(3,10))


  SMAE.int.all<-c(SMAE.int.M.Accuracy,
              SMAE.int.M.MAE,
              SMAE.int.M.MAE.int)
  
  p1<-shapiro.test(SMAE.int.M.Accuracy)$p.value
  p2<-shapiro.test(SMAE.int.M.MAE)$p.value
  p3<-shapiro.test(SMAE.int.M.MAE.int)$p.value
  
  if (p1>=0.05 & p2>=0.05 & p3>=0.05)
  {
    test.SMAE.int.less<-pairwise.t.test(SMAE.int.all,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
    test.SMAE.int.greater<-pairwise.t.test(SMAE.int.all,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
  } else {
    test.SMAE.int.less<-pairwise.wilcox.test(SMAE.int.all,Group.three,paired=TRUE,alternative="less", p.adjust.methods="holm")
    test.SMAE.int.greater<-pairwise.wilcox.test(SMAE.int.all,Group.three,paired=TRUE,alternative="greater", p.adjust.methods="holm")
  }

  
  test.SMAE.int.less
  test.SMAE.int.greater
  
  
  ##############################################################################
  #############################
  ######### knn
  ######### library(caret)
  ######### model = knnreg(medv ~ ., data = Boston)
  ######### model
  ## 5-nearest neighbor regression model


  library(caret)
 
 set.seed(12345)
 seeds<-sample(1:10000,10,replace=FALSE)
 
 pred.test.knn.reg<-list()
 test.set.rows<-list()
 
 for (i in 1:length(seeds))
 {yyy <- setdiff(1:nrow(df), random.sampl)
 set.seed(seeds[i])
 test.set.rows[[i]]<-sample(yyy,500,replace=FALSE)

   pred.test.knn.reg[[i]]<-caret::knnregTrain(train=df[random.sampl, -c(1025:1028)], test=df[test.set.rows[[i]],-c(1025:1028)],
                                              y=df[random.sampl,1025],k=5,use.all=TRUE)
     
   print(i)
 }
 
 
 cut.points<-c(0,2,10,15,35,60,1000)
 
 pred.test.knn.reg.bin<-list()
 M.knn.reg<-list()
 Accuracy.knn.reg<-vector()
 SMAE.knn.reg<-vector()
 SMAE.int.knn.reg<-vector()
 
 for (i in 1:length(seeds))
 {pred.test.knn.reg.bin[[i]]<-as.numeric(arules::discretize(pred.prueba, method = "fixed", breaks=cut.points, infinity=TRUE))
  # 
  M.knn.reg[[i]]<-table(pred.test.knn.reg.bin[[i]],df[test.set.rows[[i]],1028]) 
  #
  Accuracy.knn.reg[i]<-sum(diag(M.knn.reg[[i]]))/sum(M.knn.reg[[i]])
  #
  SMAE.knn.reg[i]<-SMAE(mat.square(M.knn.reg[[i]],levels.age.ordinal.encod))
  #
  SMAE.int.knn.reg[i]<-SMAE.int(mat.square(M.knn.reg[[i]],levels.age.ordinal.encod),Len)
}
  
 
 Accuracy.knn.reg
 SMAE.knn.reg
 SMAE.int.knn.reg
 
 mean(Accuracy.knn.reg,na.rm=TRUE)
 mean(SMAE.knn.reg,na.rm=TRUE)
 mean(SMAE.int.knn.reg,na.rm=TRUE)
 
sd(Accuracy.knn.reg,na.rm=TRUE)
sd(SMAE.knn.reg,na.rm=TRUE)
sd(SMAE.int.knn.reg,na.rm=TRUE)
 
 
 
 ##############################################################################
 ### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 ##############################################################################
 ##############################################################################
 ############# Save in a dataframe of results: metric values, for each run
 
 df.results.run.5<-data.frame(Accuracy.M.Accuracy.knn.l.0,Accuracy.M.MAE.knn.l.0,Accuracy.M.MAE.int.knn.l.0,
                              Accuracy.M.Accuracy.knn.l.1,Accuracy.M.MAE.knn.l.1,Accuracy.M.MAE.int.knn.l.1,
                              Accuracy.M.Accuracy.knn.l.2,Accuracy.M.MAE.knn.l.2,Accuracy.M.MAE.int.knn.l.2,
                              Accuracy.M.Accuracy.knn.l.3,Accuracy.M.MAE.knn.l.3,Accuracy.M.MAE.int.knn.l.3,
                              Accuracy.M.Accuracy.knn.l.4,Accuracy.M.MAE.knn.l.4,Accuracy.M.MAE.int.knn.l.4,
                              Accuracy.M.Accuracy.knn.l.5,Accuracy.M.MAE.knn.l.5,Accuracy.M.MAE.int.knn.l.5,
                              Accuracy.M.Accuracy.knn.l.6,Accuracy.M.MAE.knn.l.6,Accuracy.M.MAE.int.knn.l.6,
                              Accuracy.M.Accuracy.knn.l.7,Accuracy.M.MAE.knn.l.7,Accuracy.M.MAE.int.knn.l.7,
                              Accuracy.M.Accuracy.knn.l.8,Accuracy.M.MAE.knn.l.8,Accuracy.M.MAE.int.knn.l.8,
                              Accuracy.M.Accuracy.knn.l.9,Accuracy.M.MAE.knn.l.9,Accuracy.M.MAE.int.knn.l.9,
                              Accuracy.M.Accuracy.knn.l.10,Accuracy.M.MAE.knn.l.10,Accuracy.M.MAE.int.knn.l.10,
                              Accuracy.M.Accuracy.knn.l.11,Accuracy.M.MAE.knn.l.11,Accuracy.M.MAE.int.knn.l.11,
                              Accuracy.M.Accuracy.knn.l.12,Accuracy.M.MAE.knn.l.12,Accuracy.M.MAE.int.knn.l.12,
                              Accuracy.M.Accuracy.knn.l.13,Accuracy.M.MAE.knn.l.13,Accuracy.M.MAE.int.knn.l.13,
                              # Accuracy.M.Accuracy.knn.l.14,Accuracy.M.MAE.knn.l.14,Accuracy.M.MAE.int.knn.l.14,
                              # Accuracy.M.Accuracy.knn.l.15,Accuracy.M.MAE.knn.l.15,Accuracy.M.MAE.int.knn.l.15,
                              # Accuracy.M.Accuracy.knn.l.16,Accuracy.M.MAE.knn.l.16,Accuracy.M.MAE.int.knn.l.16,
                              ##
                              SMAE.M.Accuracy.knn.l.0,SMAE.M.MAE.knn.l.0,SMAE.M.MAE.int.knn.l.0,
                              SMAE.M.Accuracy.knn.l.1,SMAE.M.MAE.knn.l.1,SMAE.M.MAE.int.knn.l.2,
                              SMAE.M.Accuracy.knn.l.2,SMAE.M.MAE.knn.l.2,SMAE.M.MAE.int.knn.l.2,
                              SMAE.M.Accuracy.knn.l.3,SMAE.M.MAE.knn.l.3,SMAE.M.MAE.int.knn.l.3,
                              SMAE.M.Accuracy.knn.l.4,SMAE.M.MAE.knn.l.4,SMAE.M.MAE.int.knn.l.4,
                              SMAE.M.Accuracy.knn.l.5,SMAE.M.MAE.knn.l.5,SMAE.M.MAE.int.knn.l.5,
                              SMAE.M.Accuracy.knn.l.6,SMAE.M.MAE.knn.l.6,SMAE.M.MAE.int.knn.l.6,
                              SMAE.M.Accuracy.knn.l.7,SMAE.M.MAE.knn.l.7,SMAE.M.MAE.int.knn.l.7,
                              SMAE.M.Accuracy.knn.l.8,SMAE.M.MAE.knn.l.8,SMAE.M.MAE.int.knn.l.8,
                              SMAE.M.Accuracy.knn.l.9,SMAE.M.MAE.knn.l.9,SMAE.M.MAE.int.knn.l.9,
                              SMAE.M.Accuracy.knn.l.10,SMAE.M.MAE.knn.l.10,SMAE.M.MAE.int.knn.l.10,
                              SMAE.M.Accuracy.knn.l.11,SMAE.M.MAE.knn.l.11,SMAE.M.MAE.int.knn.l.11,
                              SMAE.M.Accuracy.knn.l.12,SMAE.M.MAE.knn.l.12,SMAE.M.MAE.int.knn.l.12,
                              SMAE.M.Accuracy.knn.l.13,SMAE.M.MAE.knn.l.13,SMAE.M.MAE.int.knn.l.13,
                              # SMAE.M.Accuracy.knn.l.14,SMAE.M.MAE.knn.l.14,SMAE.M.MAE.int.knn.l.14,
                              # SMAE.M.Accuracy.knn.l.15,SMAE.M.MAE.knn.l.15,SMAE.M.MAE.int.knn.l.15,
                              # SMAE.M.Accuracy.knn.l.16,SMAE.M.MAE.knn.l.16,SMAE.M.MAE.int.knn.l.16,
                              ##
                              SMAE.int.M.Accuracy.knn.l.0,SMAE.int.M.MAE.knn.l.0,SMAE.int.M.MAE.int.knn.l.0,
                              SMAE.int.M.Accuracy.knn.l.1,SMAE.int.M.MAE.knn.l.1,SMAE.int.M.MAE.int.knn.l.1,
                              SMAE.int.M.Accuracy.knn.l.2,SMAE.int.M.MAE.knn.l.2,SMAE.int.M.MAE.int.knn.l.2,
                              SMAE.int.M.Accuracy.knn.l.3,SMAE.int.M.MAE.knn.l.3,SMAE.int.M.MAE.int.knn.l.3,
                              SMAE.int.M.Accuracy.knn.l.4,SMAE.int.M.MAE.knn.l.4,SMAE.int.M.MAE.int.knn.l.4,
                              SMAE.int.M.Accuracy.knn.l.5,SMAE.int.M.MAE.knn.l.5,SMAE.int.M.MAE.int.knn.l.5,
                              SMAE.int.M.Accuracy.knn.l.6,SMAE.int.M.MAE.knn.l.6,SMAE.int.M.MAE.int.knn.l.6,
                              SMAE.int.M.Accuracy.knn.l.7,SMAE.int.M.MAE.knn.l.7,SMAE.int.M.MAE.int.knn.l.7,
                              SMAE.int.M.Accuracy.knn.l.8,SMAE.int.M.MAE.knn.l.8,SMAE.int.M.MAE.int.knn.l.8,
                              SMAE.int.M.Accuracy.knn.l.9,SMAE.int.M.MAE.knn.l.9,SMAE.int.M.MAE.int.knn.l.9,
                              SMAE.int.M.Accuracy.knn.l.10,SMAE.int.M.MAE.knn.l.10,SMAE.int.M.MAE.int.knn.l.10,
                              SMAE.int.M.Accuracy.knn.l.11,SMAE.int.M.MAE.knn.l.11,SMAE.int.M.MAE.int.knn.l.11,
                              SMAE.int.M.Accuracy.knn.l.12,SMAE.int.M.MAE.knn.l.12,SMAE.int.M.MAE.int.knn.l.12,
                              SMAE.int.M.Accuracy.knn.l.13,SMAE.int.M.MAE.knn.l.13,SMAE.int.M.MAE.int.knn.l.13,
                              # SMAE.int.M.Accuracy.knn.l.14,SMAE.int.M.MAE.knn.l.14,SMAE.int.M.MAE.int.knn.l.14,
                              # SMAE.int.M.Accuracy.knn.l.15,SMAE.int.M.MAE.knn.l.15,SMAE.int.M.MAE.int.knn.l.15,
                              # SMAE.int.M.Accuracy.knn.l.16,SMAE.int.M.MAE.knn.l.16,SMAE.int.M.MAE.int.knn.l.16,
                              ##
                              Accuracy.knn.reg,
                              SMAE.knn.reg,
                              SMAE.int.knn.reg
 )
                              
 str(df.results.run.5) 
 View(df.results.run.5)


 save(df.results.run.5, file = "results_knn_run_5.RData")
 write.csv(df.results.run.5,"results_knn_run_5.csv", row.names = TRUE)
                               
  

 
 ################################################################################
 ################################################################################
 ################################################################################
 ################################################################################
 ################################################################################
 ################################################################################
 ################################################################################
 
 
 
 
 
 
 ################################################################################
 ################################################################################
 ########## e1071::tune.knn. Resampling method: bootstrap
 ################################################################################
 
 tuned.knn.Accuracy.boot <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                   k=1:20,
                                   tunecontrol = tune.control(sampling = "boot", error.fun=NULL))
 
 
 tuned.knn.MAE.boot <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                   k=1:20,
                                   tunecontrol = tune.control(sampling = "boot", error.fun=standard.MAE.ord))
 
 
 tuned.knn.MAE.int.boot <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                       k=1:20, 
                                       tunecontrol = tune.control(sampling = "boot",error.fun=standard.MAE.int))
 
 
 
 summary(tuned.knn.Accuracy.boot)
 plot(tuned.knn.Accuracy.boot)
 tuned.knn.Accuracy.boot$best.parameters
 
 
 summary(tuned.knn.MAE.boot)
 plot(tuned.knn.MAE.boot)
 tuned.knn.MAE.boot$best.parameters
 
 
 summary(tuned.knn.MAE.int.boot)
 plot(tuned.knn.MAE.int.boot)
 tuned.knn.MAE.int.boot$best.parameters
 
 
 
 ################################################################################
 ################################################################################
 ########## e1071::tune.knn. Resampling method: fix
 ################################################################################
 
 tuned.knn.Accuracy.fix <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                             k=1:20,
                                             tunecontrol = tune.control(sampling = "fix", error.fun=NULL))
 
 
 tuned.knn.MAE.fix <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                        k=1:20,
                                        tunecontrol = tune.control(sampling = "fix", error.fun=standard.MAE.ord))
 
 
 tuned.knn.MAE.int.fix <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                            k=1:20, 
                                            tunecontrol = tune.control(sampling = "fix",error.fun=standard.MAE.int))
 
 
 
 summary(tuned.knn.Accuracy.fix)
 plot(tuned.knn.Accuracy.fix)
 tuned.knn.Accuracy.fix$best.parameters
 
 
 summary(tuned.knn.MAE.fix)
 plot(tuned.knn.MAE.fix)
 tuned.knn.MAE.fix$best.parameters
 
 
 summary(tuned.knn.MAE.int.fix)
 plot(tuned.knn.MAE.int.fix)
 tuned.knn.MAE.int.fix$best.parameters
 
 
 
 
 l.max<-min(tuned.knn.Accuracy.fix$best.parameters,tuned.knn.MAE.fix$best.parameters,tuned.knn.MAE.int.fix$best.parameters)
 
 r=0:l.max
 
 h=5
 
 

 
 library(class)
 
 set.seed(12345)
 seeds<-sample(1:10000,10,replace=FALSE)
 
 pred.test.knn.Accuracy.fix<-list()
 pred.test.knn.MAE.fix<-list()
 pred.test.knn.MAE.int.fix<-list()
 test.set.rows<-list()
 
 for (i in 1:length(seeds))
 {set.seed(seeds[i])
   random.sampl<-sample(1:(dim(df)[1]),2000,replace=FALSE)
   yyy <- setdiff(1:nrow(df), random.sampl)
   set.seed(seeds[i])
   test.set.rows[[i]]<-sample(yyy,500,replace=FALSE)
   
   pred.test.knn.Accuracy.fix[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
                                           k=tuned.knn.Accuracy.fix$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
   
   pred.test.knn.MAE.fix[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
                                          k=tuned.knn.MAE.fix$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
   
   pred.test.knn.MAE.int.fix[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
                                          k=tuned.knn.MAE.int.fix$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
   
   print(i)
 }
 
 
 

 ################################################################################
 ################################################################################
 ################## Confusion matrices, Accuracy, SMAE and SMAE.int
 ################################################################################
 
 M.Accuracy.fix<-list()
 M.MAE.fix<-list()
 M.MAE.int.fix<-list()
 #
 Accuracy.M.Accuracy.fix<-vector()
 Accuracy.M.MAE.fix<-vector()
 Accuracy.M.MAE.int.fix<-vector() 
 #
 SMAE.M.Accuracy.fix<-vector()
 SMAE.M.MAE.fix<-vector()
 SMAE.M.MAE.int.fix<-vector()
 #
 SMAE.int.M.Accuracy.fix<-vector()
 SMAE.int.M.MAE.fix<-vector()
 SMAE.int.M.MAE.int.fix<-vector()
 
 for (i in 1:length(seeds))
 {M.Accuracy.fix[[i]]<-table(pred.test.knn.Accuracy.fix[[i]],df[test.set.rows[[i]],1028])
 M.MAE.fix[[i]]<-table(pred.test.knn.MAE.fix[[i]],df[test.set.rows[[i]],1028])
 M.MAE.int.fix[[i]]<-table(pred.test.knn.MAE.int.fix[[i]],df[test.set.rows[[i]],1028])
 ##
 Accuracy.M.Accuracy.fix[i]<-sum(diag(M.Accuracy.fix[[i]]))/sum(M.Accuracy.fix[[i]])
 Accuracy.M.MAE.fix[i]<-sum(diag(M.MAE.fix[[i]]))/sum(M.MAE.fix[[i]])
 Accuracy.M.MAE.int.fix[i]<-sum(diag(M.MAE.int.fix[[i]]))/sum(M.MAE.int.fix[[i]])
 ##
 SMAE.M.Accuracy.fix[i]<-SMAE(mat.square(M.Accuracy.fix[[i]],levels.age.ordinal.encod))
 SMAE.M.MAE.fix[i]<-SMAE(mat.square(M.MAE.fix[[i]],levels.age.ordinal.encod))
 SMAE.M.MAE.int.fix[i]<-SMAE(mat.square(M.MAE.int.fix[[i]],levels.age.ordinal.encod))
 ##
 SMAE.int.M.Accuracy.fix[i]<-SMAE.int(mat.square(M.Accuracy.fix[[i]],levels.age.ordinal.encod),Len)
 SMAE.int.M.MAE.fix[i]<-SMAE.int(mat.square(M.MAE.fix[[i]],levels.age.ordinal.encod),Len)
 SMAE.int.M.MAE.int.fix[i]<-SMAE.int(mat.square(M.MAE.int.fix[[i]],levels.age.ordinal.encod),Len)
 }
 
 Accuracy.M.Accuracy.fix
 Accuracy.M.MAE.fix
 Accuracy.M.MAE.int.fix
 
 mean(Accuracy.M.Accuracy.fix,na.rm=TRUE)
 mean(Accuracy.M.MAE.fix,na.rm=TRUE)
 mean(Accuracy.M.MAE.int.fix,na.rm=TRUE)
 
 
 
 SMAE.M.Accuracy.fix
 SMAE.M.MAE.fix
 SMAE.M.MAE.int.fix
 
 mean(SMAE.M.Accuracy.fix,na.rm=TRUE)
 mean(SMAE.M.MAE.fix,na.rm=TRUE)
 mean(SMAE.M.MAE.int.fix,na.rm=TRUE)
 
 
 SMAE.int.M.Accuracy.fix
 SMAE.int.M.MAE.fix
 SMAE.int.M.MAE.int.fix
 
 mean(SMAE.int.M.Accuracy.fix,na.rm=TRUE)
 mean(SMAE.int.M.MAE.fix,na.rm=TRUE)
 mean(SMAE.int.M.MAE.int.fix,na.rm=TRUE)
 
 
 
 
 
 ################################################################################
 ################################################################################
 ########## e1071::tune.gknn. Resampling method: cross-validation, distance=Euclidean
 ################################################################################
 
 tuned.gknn.Accuracy <- e1071::tune.knn(x = df[random.sampl ,-c(1025:1028)], y = df[random.sampl ,1028], 
                                       k=1:20, distance="Euclidean",
                                       tunecontrol = tune.control(sampling = "cross", cross=5, error.fun=NULL))
 
 tuned.gknn.MAE <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                   k=1:20,distance="Euclidean",
                                   tunecontrol = tune.control(sampling = "cross", cross=5, error.fun=standard.MAE.ord))
 
 
 tuned.gknn.MAE.int <- e1071::tune.knn (x = df[random.sampl,-c(1025:1028)], y = df[random.sampl,1028], 
                                       k=1:20,distance="Euclidean",
                                       tunecontrol = tune.control(sampling = "cross",cross=5, error.fun=standard.MAE.int))
 
 
 
 
 
 summary(tuned.gknn.Accuracy)
 plot(tuned.gknn.Accuracy)
 tuned.gknn.Accuracy$best.parameters
 
 
 summary(tuned.gknn.MAE)
 plot(tuned.gknn.MAE)
 tuned.gknn.MAE$best.parameters
 
 
 summary(tuned.gknn.MAE.int)
 plot(tuned.gknn.MAE.int)
 tuned.gknn.MAE.int$best.parameters
 
 
 
 l.max<-min(tuned.gknn.Accuracy$best.parameters,tuned.gknn.MAE$best.parameters,tuned.gknn.MAE.int$best.parameters)
 
 r=0:l.max
 
 h=6
 
 
 
 
 library(class)
 
 set.seed(12345)
 seeds<-sample(1:10000,10,replace=FALSE)
 
 pred.test.gknn.Accuracy<-list()
 pred.test.gknn.MAE<-list()
 pred.test.gknn.MAE.int<-list()
 test.set.rows<-list()
 
 for (i in 1:length(seeds))
 {set.seed(seeds[i])
   random.sampl<-sample(1:(dim(df)[1]),2000,replace=FALSE)
   yyy <- setdiff(1:nrow(df), random.sampl)
   set.seed(seeds[i])
   test.set.rows[[i]]<-sample(yyy,500,replace=FALSE)
   
   pred.test.gknn.Accuracy[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
                                               k=tuned.gknn.Accuracy$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
   
   pred.test.gknn.MAE[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
                                          k=tuned.gknn.MAE$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
   
   pred.test.gknn.MAE.int[[i]]<-class::knn(train=df[random.sampl, -c(1025:1027)], test=df[test.set.rows[[i]],-c(1025:1027)],cl=df[random.sampl,1028],
                                              k=tuned.gknn.MAE.int$best.parameters,l=min(max(r[h],0),l.max),prob=FALSE, use.all=TRUE)
   
   print(i)
 }
 
 

 ################################################################################
 ################################################################################
 ################## Confusion matrices, Accuracy, SMAE and SMAE.int
 ################################################################################
 
 M.Accuracy.gknn<-list()
 M.MAE.gknn<-list()
 M.MAE.int.gknn<-list()
 #
 Accuracy.M.Accuracy.gknn<-vector()
 Accuracy.M.MAE.gknn<-vector()
 Accuracy.M.MAE.int.gknn<-vector() 
 #
 SMAE.M.Accuracy.gknn<-vector()
 SMAE.M.MAE.gknn<-vector()
 SMAE.M.MAE.int.gknn<-vector()
 #
 SMAE.int.M.Accuracy.gknn<-vector()
 SMAE.int.M.MAE.gknn<-vector()
 SMAE.int.M.MAE.int.gknn<-vector()
 
 for (i in 1:length(seeds))
 {M.Accuracy.gknn[[i]]<-table(pred.test.gknn.Accuracy[[i]],df[test.set.rows[[i]],1028])
 M.MAE.gknn[[i]]<-table(pred.test.gknn.MAE[[i]],df[test.set.rows[[i]],1028])
 M.MAE.int.gknn[[i]]<-table(pred.test.gknn.MAE.int[[i]],df[test.set.rows[[i]],1028])
 ##
 Accuracy.M.Accuracy.gknn[i]<-sum(diag(M.Accuracy.gknn[[i]]))/sum(M.Accuracy.gknn[[i]])
 Accuracy.M.MAE.gknn[i]<-sum(diag(M.MAE.gknn[[i]]))/sum(M.MAE.gknn[[i]])
 Accuracy.M.MAE.int.gknn[i]<-sum(diag(M.MAE.int.gknn[[i]]))/sum(M.MAE.int.gknn[[i]])
 ##
 SMAE.M.Accuracy.gknn[i]<-SMAE(mat.square(M.Accuracy.gknn[[i]],levels.age.ordinal.encod))
 SMAE.M.MAE.gknn[i]<-SMAE(mat.square(M.MAE.gknn[[i]],levels.age.ordinal.encod))
 SMAE.M.MAE.int.gknn[i]<-SMAE(mat.square(M.MAE.int.gknn[[i]],levels.age.ordinal.encod))
 ##
 SMAE.int.M.Accuracy.gknn[i]<-SMAE.int(mat.square(M.Accuracy.gknn[[i]],levels.age.ordinal.encod),Len)
 SMAE.int.M.MAE.gknn[i]<-SMAE.int(mat.square(M.MAE.gknn[[i]],levels.age.ordinal.encod),Len)
 SMAE.int.M.MAE.int.gknn[i]<-SMAE.int(mat.square(M.MAE.int.gknn[[i]],levels.age.ordinal.encod),Len)
 }
 
 Accuracy.M.Accuracy.gknn
 Accuracy.M.MAE.gknn
 Accuracy.M.MAE.int.gknn
 
 mean(Accuracy.M.Accuracy.gknn,na.rm=TRUE)
 mean(Accuracy.M.MAE.gknn,na.rm=TRUE)
 mean(Accuracy.M.MAE.int.gknn,na.rm=TRUE)
 
 
 
 SMAE.M.Accuracy.gknn
 SMAE.M.MAE.gknn
 SMAE.M.MAE.int.gknn
 
 mean(SMAE.M.Accuracy.gknn,na.rm=TRUE)
 mean(SMAE.M.MAE.gknn,na.rm=TRUE)
 mean(SMAE.M.MAE.int.gknn,na.rm=TRUE)
 
 
 SMAE.int.M.Accuracy.gknn
 SMAE.int.M.MAE.gknn
 SMAE.int.M.MAE.int.gknn
 
 mean(SMAE.int.M.Accuracy.gknn,na.rm=TRUE)
 mean(SMAE.int.M.MAE.gknn,na.rm=TRUE)
 mean(SMAE.int.M.MAE.int.gknn,na.rm=TRUE)
 
 
 
 
 
 
 
 
 ####################################################
##############. DE MOMENTO, KERAS NO!!!!!!!!!!!!!!!






# install_keras()

library(keras)
library(tensorflow)
library(reticulate)


model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(9673)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 5, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


x_train<-training[[1]][,c(feat)]

###

y_train<-training[[1]]$age.bin

y_train_2 <- as.numeric(y_train)

y_train_3 <- as.integer(y_train_2)-1

summary(y_train_3)
  
y_train_4<-to_categorical(y_train_3, num_classes=5)
y_train_4




history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 130, 
  validation_split = 0.2
)



#################################








tune.control <- function(random = FALSE,
                         nrepeat = 1,
                         repeat.aggregate = mean,
                         sampling = c("cross", "fix", "bootstrap"),
                         sampling.aggregate = mean,
                         sampling.dispersion = sd,
                         cross = 10,
                         fix = 2 / 3,
                         nboot = 10,
                         boot.size = 9 / 10,
                         best.model = TRUE,
                         performances = TRUE,
                         error.fun = NULL) {
  structure(list(random = random,
                 nrepeat = nrepeat,
                 repeat.aggregate = repeat.aggregate,
                 sampling = match.arg(sampling),
                 sampling.aggregate = sampling.aggregate,
                 sampling.dispersion = sampling.dispersion,
                 cross = cross,
                 fix = fix,
                 nboot = nboot,
                 boot.size = boot.size,
                 best.model = best.model,
                 performances = performances,
                 error.fun = error.fun
  ),
  class = "tune.control"
  )
}

tune <- function(METHOD, train.x, train.y = NULL, data = list(),
                 validation.x = NULL, validation.y = NULL,
                 ranges = NULL, predict.func = predict,
                 tunecontrol = tune.control(),
                 ...
) {
  call <- match.call()
  
  ## internal helper functions
  resp <- function(formula, data) {
    
    model.response(model.frame(formula, data))
  }
  
  classAgreement <- function (tab) {
    n <- sum(tab)
    if (!is.null(dimnames(tab))) {
      lev <- intersect(colnames(tab), rownames(tab))
      p0 <- sum(diag(tab[lev, lev])) / n
    } else {
      m <- min(dim(tab))
      p0 <- sum(diag(tab[1:m, 1:m])) / n
    }
    p0
  }
  
  ## parameter handling
  if (tunecontrol$sampling == "cross")
    validation.x <- validation.y <- NULL
  useFormula <- is.null(train.y)
  if (useFormula && (is.null(data) || length(data) == 0))
    data <- model.frame(train.x)
  if (is.vector(train.x)) train.x <- t(t(train.x))
  if (is.data.frame(train.y))
    train.y <- as.matrix(train.y)
  
  ## prepare training indices
  if (!is.null(validation.x)) tunecontrol$fix <- 1
  n <- nrow(if (useFormula) data else train.x)
  perm.ind <- sample(n)
  if (tunecontrol$sampling == "cross") {
    if (tunecontrol$cross > n)
      stop(sQuote("cross"), " must not exceed sampling size!")
    if (tunecontrol$cross == 1)
      stop(sQuote("cross"), " must be greater than 1!")
  }
  train.ind <- if (tunecontrol$sampling == "cross")
    tapply(1:n, cut(1:n, breaks = tunecontrol$cross), function(x) perm.ind[-x])
  else if (tunecontrol$sampling == "fix")
    list(perm.ind[1:trunc(n * tunecontrol$fix)])
  else ## bootstrap
    lapply(1:tunecontrol$nboot,
           function(x) sample(n, n * tunecontrol$boot.size, replace = TRUE))
  
  ## find best model
  parameters <- if (is.null(ranges))
    {data.frame(dummyparameter = 0)
  } else 
    {expand.grid(ranges)}
  
  p <- nrow(parameters)
  if (!is.logical(tunecontrol$random)) {
    if (tunecontrol$random < 1)
      stop("random must be a strictly positive integer")
    if (tunecontrol$random > p) tunecontrol$random <- p
    parameters <- parameters[sample(1:p, tunecontrol$random),]
    p <- nrow(parameters)
  }
  model.variances <- model.errors <- c()
  
  ## - loop over all models
  for (para.set in 1:p) {
    sampling.errors <- c()
    
    ## - loop over all training samples
    for (sample in 1:length(train.ind)) {
      repeat.errors <- c()
      
      ## - repeat training `nrepeat' times
      for (reps in 1:tunecontrol$nrepeat) {
        
        ## train one model
        pars <- if (is.null(ranges))
          NULL
        else
          lapply(parameters[para.set,,drop = FALSE], unlist)
        
        model <- if (useFormula)
          do.call(METHOD, c(list(train.x,
                                 data = data,
                                 subset = train.ind[[sample]]),
                            pars, list(...)
          )
          )
        else
          do.call(METHOD, c(list(train.x[train.ind[[sample]],],
                                 y = train.y[train.ind[[sample]]]),
                            pars, list(...)
          )
          )
        
        ## predict validation set
        pred <- predict.func(model,
                             if (!is.null(validation.x))
                               validation.x
                             else if (useFormula)
                               data[-train.ind[[sample]],,drop = FALSE]
                             else if (inherits(train.x, "matrix.csr"))
                               train.x[-train.ind[[sample]],]
                             else
                               train.x[-train.ind[[sample]],,drop = FALSE]
        )
        
        ## compute performance measure
        true.y <- if (!is.null(validation.y))
          validation.y
        else if (useFormula) {
          if (!is.null(validation.x))
            resp(train.x, validation.x)
          else
            resp(train.x, data[-train.ind[[sample]],])
        } else
          train.y[-train.ind[[sample]]]
        
        if (is.null(true.y)) true.y <- rep(TRUE, length(pred))
      
        repeat.errors[reps] <- if (!is.null(tunecontrol$error.fun))
          tunecontrol$error.fun(true.y, pred)
        else if ((is.logical(true.y) || is.factor(true.y)) && (is.logical(pred) || is.factor(pred) || is.character(pred))) ## classification error
          1 - classAgreement(table(pred, true.y))
        else if (ordinal==FALSE && is.numeric(true.y) && is.numeric(pred)) ## MSE (regression)
          crossprod(pred - true.y) / length(pred)
        else
          stop("Dependent variable has wrong type!")
      }
      sampling.errors[sample] <- tunecontrol$repeat.aggregate(repeat.errors)
    }
    model.errors[para.set] <- tunecontrol$sampling.aggregate(sampling.errors)
    model.variances[para.set] <- tunecontrol$sampling.dispersion(sampling.errors)
  }
  
  ## return results
  best <- which.min(model.errors)
  pars <- if (is.null(ranges))
    NULL
  else
    lapply(parameters[best,,drop = FALSE], unlist)
  structure(list(best.parameters  = parameters[best,,drop = FALSE],
                 best.performance = model.errors[best],
                 method           = if (!is.character(METHOD))
                   deparse(substitute(METHOD)) else METHOD,
                 nparcomb         = nrow(parameters),
                 train.ind        = train.ind,
                 sampling         = switch(tunecontrol$sampling,
                                           fix = "fixed training/validation set",
                                           bootstrap = "bootstrapping",
                                           cross = if (tunecontrol$cross == n) "leave-one-out" else
                                             paste(tunecontrol$cross,"-fold cross validation", sep="")
                 ),
                 performances     = if (tunecontrol$performances) cbind(parameters, error = model.errors, dispersion = model.variances),
                 best.model       = if (tunecontrol$best.model) {
                   modeltmp <- if (useFormula)
                     do.call(METHOD, c(list(train.x, data = data),
                                       pars, list(...)))
                   else
                     do.call(METHOD, c(list(x = train.x,
                                            y = train.y),
                                       pars, list(...)))
                   call[[1]] <- as.symbol("best.tune")
                   modeltmp$call <- call
                   modeltmp
                 }
  ),
  class = "tune"
  )
}

best.tune <- function(...) {
  call <- match.call()
  modeltmp <- tune(...)$best.model
  modeltmp$call <- call
  modeltmp
}

print.tune <- function(x, ...) {
  if (x$nparcomb > 1) {
    cat("\nParameter tuning of ", sQuote(x$method), ":\n\n", sep="")
    cat("- sampling method:", x$sampling,"\n\n")
    cat("- best parameters:\n")
    tmp <- x$best.parameters
    rownames(tmp) <- ""
    print(tmp)
    cat("\n- best performance:", x$best.performance, "\n")
    cat("\n")
  } else {
    cat("\nError estimation of ", sQuote(x$method), " using ", x$sampling, ": ",
        x$best.performance, "\n\n", sep="")
  }
}

summary.tune <- function(object, ...)
  structure(object, class = "summary.tune")

print.summary.tune <- function(x, ...) {
  print.tune(x)
  if (!is.null(x$performances) && (x$nparcomb > 1)) {
    cat("- Detailed performance results:\n")
    print(x$performances)
    cat("\n")
  }
}

hsv_palette <- function(h = 2/3, from = 0.7, to = 0.2, v = 1)
  function(n) hsv(h = h, s = seq(from, to, length.out = n), v = v)

plot.tune <- function(x,
                      type=c("contour","perspective"),
                      theta=60,
                      col="lightblue",
                      main = NULL,
                      xlab = NULL,
                      ylab = NULL,
                      swapxy = FALSE,
                      transform.x = NULL,
                      transform.y = NULL,
                      transform.z = NULL,
                      color.palette = hsv_palette(),
                      nlevels = 20,
                      ...)
{
  if (is.null(x$performances))
    stop("Object does not contain detailed performance measures!")
  k <- ncol(x$performances)
  if (k > 4) stop("Cannot visualize more than 2 parameters")
  type = match.arg(type)
  
  if (is.null(main))
    main <- paste("Performance of `", x$method, "'", sep="")
  
  if (k == 3)
    plot(x$performances[,1:2], type = "b", main = main)
  else  {
    if (!is.null(transform.x))
      x$performances[,1] <- transform.x(x$performances[,1])
    if (!is.null(transform.y))
      x$performances[,2] <- transform.y(x$performances[,2])
    if (!is.null(transform.z))
      x$performances[,3] <- transform.z(x$performances[,3])
    if (swapxy)
      x$performances[,1:2] <- x$performances[,2:1]
    x <- xtabs(error~., data = x$performances[,-k])
    if (is.null(xlab)) xlab <- names(dimnames(x))[1 + swapxy]
    if (is.null(ylab)) ylab <- names(dimnames(x))[2 - swapxy]
    if (type == "perspective")
      persp(x=as.double(rownames(x)),
            y=as.double(colnames(x)),
            z=x,
            xlab=xlab,
            ylab=ylab,
            zlab="accuracy",
            theta=theta,
            col=col,
            ticktype="detailed",
            main = main,
            ...
      )
    else
      filled.contour(x=as.double(rownames(x)),
                     y=as.double(colnames(x)),
                     xlab=xlab,
                     ylab=ylab,
                     nlevels=nlevels,
                     color.palette = color.palette,
                     main = main,
                     x, ...)
  }
}

#############################################
## convenience functions for some methods
#############################################

tune.svm <- function(x, y = NULL, data = NULL, degree = NULL, gamma = NULL,
                     coef0 = NULL, cost = NULL, nu = NULL, class.weights = NULL,
                     epsilon = NULL, ...) {
  call <- match.call()
  call[[1]] <- as.symbol("best.svm")
  ranges <- list(degree = degree, gamma = gamma,
                 coef0 = coef0, cost = cost, nu = nu,
                 class.weights = class.weights, epsilon = epsilon)
  ranges[vapply(ranges, is.null, NA)] <- NULL
  if (length(ranges) < 1)
    ranges = NULL
  modeltmp <- if (inherits(x, "formula"))
    tune("svm", train.x = x, data = data, ranges = ranges, ...)
  else
    tune("svm", train.x = x, train.y = y, ranges = ranges, ...)
  if (!is.null(modeltmp$best.model))
    modeltmp$best.model$call <- call
  modeltmp
}

best.svm <- function(x, tunecontrol = tune.control(), ...) {
  call <- match.call()
  tunecontrol$best.model = TRUE
  modeltmp <- tune.svm(x, ..., tunecontrol = tunecontrol)$best.model
  modeltmp$call <- call
  modeltmp
}

tune.nnet <- function(x, y = NULL, data = NULL,
                      size = NULL, decay = NULL, trace = FALSE,
                      tunecontrol = tune.control(nrepeat = 5),
                      ...) {
  call <- match.call()
  call[[1]] <- as.symbol("best.nnet")
  loadNamespace("nnet")
  predict.func <- predict
  useFormula <- inherits(x, "formula")
  if (is.factor(y) ||
      (useFormula && is.factor(model.response(model.frame(formula = x, data = data))))
  )
    predict.func = function(...) predict(..., type = "class")
  ranges <- list(size = size, decay = decay)
  ranges[vapply(ranges, is.null, NA)] <- NULL
  if (length(ranges) < 1)
    ranges = NULL
  modeltmp <- if (useFormula)
    tune("nnet", train.x = x, data = data, ranges = ranges, predict.func = predict.func,
         tunecontrol = tunecontrol, trace = trace, ...)
  else
    tune("nnet", train.x = x, train.y = y, ranges = ranges, predict.func = predict.func,
         tunecontrol = tunecontrol, trace = trace, ...)
  if (!is.null(modeltmp$best.model))
    modeltmp$best.model$call <- call
  modeltmp
}

best.nnet <- function(x, tunecontrol = tune.control(nrepeat = 5), ...) {
  call <- match.call()
  tunecontrol$best.model = TRUE
  modeltmp <- tune.nnet(x, ..., tunecontrol = tunecontrol)$best.model
  modeltmp$call <- call
  modeltmp
}

tune.randomForest <- function(x, y = NULL, data = NULL, nodesize = NULL, mtry = NULL, ntree = NULL, ...) {
  call <- match.call()
  call[[1]] <- as.symbol("best.randomForest")
  loadNamespace("randomForest")
  ranges <- list(nodesize = nodesize, mtry = mtry, ntree = ntree)
  ranges[vapply(ranges, is.null, NA)] <- NULL
  if (length(ranges) < 1)
    ranges = NULL
  modeltmp <- if (inherits(x, "formula"))
    tune("randomForest", train.x = x, data = data, ranges = ranges, ...)
  else
    tune("randomForest", train.x = x, train.y = y, ranges = ranges, ...)
  if (!is.null(modeltmp$best.model))
    modeltmp$best.model$call <- call
  modeltmp
}

best.randomForest <- function(x, tunecontrol = tune.control(), ...) {
  call <- match.call()
  tunecontrol$best.model = TRUE
  modeltmp <- tune.randomForest(x, ..., tunecontrol = tunecontrol)$best.model
  modeltmp$call <- call
  modeltmp
}

tune.gknn <- function(x, y = NULL, data = NULL, k = NULL, ...) {
  call <- match.call()
  call[[1]] <- as.symbol("best.gknn")
  ranges <- list(k = k)
  ranges[vapply(ranges, is.null, NA)] <- NULL
  if (length(ranges) < 1)
    ranges = NULL
  modeltmp <- if (inherits(x, "formula"))
    tune("gknn", train.x = x, data = data, ranges = ranges, ...)
  else
    tune("gknn", train.x = x, train.y = y, ranges = ranges, ...)
  if (!is.null(modeltmp$best.model))
    modeltmp$best.model$call <- call
  modeltmp
}

best.gknn <- function(x, tunecontrol = tune.control(), ...) {
  call <- match.call()
  tunecontrol$best.model = TRUE
  modeltmp <- tune.gknn(x, ..., tunecontrol = tunecontrol)$best.model
  modeltmp$call <- call
  modeltmp
}



knn.wrapper <- function(x, y, k = 1, l = 0, ...)
  list(train = x, cl = y, k = k, l = l, ...)

tune.knn <- function(x, y, k = NULL, l = NULL, ...) {
  loadNamespace("class")
  ranges <- list(k = k, l = l)
  ranges[vapply(ranges, is.null, NA)] <- NULL
  if (length(ranges) < 1)
    ranges = NULL
  tune("knn.wrapper",
       train.x = x, train.y = y, ranges = ranges,
       predict.func = function(x, ...) knn(train = x$train, cl = x$cl, k = x$k, l = x$l, ...),
       ...)
}

rpart.wrapper <- function(formula, minsplit=20, minbucket=round(minsplit/3), cp=0.01,
                          maxcompete=4, maxsurrogate=5, usesurrogate=2, xval=10,
                          surrogatestyle=0, maxdepth=30, ...)
  rpart::rpart(formula,
               control = rpart::rpart.control(minsplit=minsplit, minbucket=minbucket,
                                              cp=cp, maxcompete=maxcompete, maxsurrogate=maxsurrogate,
                                              usesurrogate=usesurrogate, xval=xval,
                                              surrogatestyle=surrogatestyle, maxdepth=maxdepth),
               ...
  )

tune.rpart <- function(formula, data, na.action = na.omit,
                       minsplit=NULL, minbucket=NULL, cp=NULL,
                       maxcompete=NULL, maxsurrogate=NULL, usesurrogate=NULL, xval=NULL,
                       surrogatestyle=NULL, maxdepth=NULL,
                       predict.func = NULL,
                       ...) {
  call <- match.call()
  call[[1]] <- as.symbol("best.rpart")
  loadNamespace("rpart")
  ranges <- list(minsplit=minsplit, minbucket=minbucket, cp=cp,
                 maxcompete=maxcompete, maxsurrogate=maxsurrogate,
                 usesurrogate=usesurrogate, xval=xval,
                 surrogatestyle=surrogatestyle, maxdepth=maxdepth)
  ranges[vapply(ranges, is.null, NA)] <- NULL
  if (length(ranges) < 1)
    ranges <- NULL
  
  predict.func <- if (is.factor(model.response(model.frame(formula, data))))
    function(...) predict(..., type = "class")
  else
    predict
  modeltmp <- tune("rpart.wrapper", train.x = formula, data = data, ranges = ranges,
                   predict.func = predict.func, na.action = na.action, ...)
  if (!is.null(modeltmp$best.model))
    modeltmp$best.model$call <- call
  modeltmp
}

best.rpart <- function(formula, tunecontrol = tune.control(), ...) {
  call <- match.call()
  tunecontrol$best.model = TRUE
  modeltmp <- tune.rpart(formula, ..., tunecontrol = tunecontrol)$best.model
  modeltmp$call <- call
  modeltmp
}
