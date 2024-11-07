###########################################
###########################################
#
#  EXPERIMENTAL PHASE (Section 5)
#
# Tuning knn with e1071
#
# Data set: Abalone
#
#
#
#
########################################
############   tune.control function, for e1071 library
############
############ https://github.com/cran/e1071/blob/master/R/tune.R
###########
########################################



source("mat_square.R")

library(arules)   # for "discretize".
library(doParallel)
registerDoParallel(cores=6)

abalone <- read.csv("abalone.data", header=FALSE)
View(abalone)
str(abalone)



# #####  variable "V9" are Rings (+1.5 gives the age in years) binning to "age.bin", with 5 intervals
# 

cut.points<-c(0,8,10,11,14,1000)

V9.bin <- arules::discretize(abalone$V9, method = "fixed", breaks=cut.points, infinity=TRUE)
table(V9.bin)
abalone<-as.data.frame(cbind(abalone,V9.bin))

abalone$V9.bin<-as.factor(abalone$V9.bin)

levels.V9.int<-names(table(abalone$V9.bin))
levels(abalone$V9.bin)<-c("<8","[8,10)","[10,11)","[11,14)",">=14")
levels.V9.ordinal.encod<-unique(as.numeric(abalone$V9.bin))

par(mfrow = c(1, 1))
plot(abalone$V9.bin,abalone$V9)


levels.V9.ordinal.encod<-sort(unique(as.numeric(abalone$V9.bin)))
abalone$V9.bin.num<-as.numeric(abalone$V9.bin)
table(abalone$V9.bin)

abalone$V9.bin.num.factor<-as.factor(abalone$V9.bin.num)   # necesitamos las clases en numero pero tipo factor

str(abalone)
table(abalone$V1)

## Knn needs to have all features numeric: introduce dummies for V1=Sex variable (with 3 values)

library(fastDummies)
yyy<-dummy_cols(abalone, select_columns = c("V1")) # V1_F column 14
                                                   # V1_I column 15
                                                   # V1_M column 16
abalone<-as.data.frame(cbind(abalone,yyy[,c(14:16)]))

################################################################################
####### Error functions for tune.control argument, tune function
#######

standard.MAE.ord.abalone<-function(y,z)   # y = true, z = predicted
{Conf.mat<- mat.square(table(as.numeric(z),as.numeric(y)),levels.V9.ordinal.encod)
value<-SMAE(Conf.mat)
value
}


#########
# intervals endpoints (assume the last one is 20)
v.20<-cut.points<-c(0,8,10,11,14,20)

Len.20<-leng(v.20)  # intervals lengths

standard.MAE.int.20.abalone<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V9.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.20)
value
}
  
#########
# intervals endpoints (assume the last one is 25)
v.25<-cut.points<-c(0,8,10,11,14,25)

Len.25<-leng(v.25)  # intervals lengths

standard.MAE.int.25.abalone<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V9.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.25)
value
}

#########
# intervals endpoints (assume the last one is 30)
v.30<-cut.points<-c(0,8,10,11,14,30)

Len.30<-leng(v.30)  # intervals lengths

standard.MAE.int.30.abalone<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V9.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.30)
value
}


#########
# intervals endpoints (assume the last one is 35)
v.35<-cut.points<-c(0,8,10,11,14,35)

Len.35<-leng(v.35)  # intervals lengths

standard.MAE.int.35.abalone<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V9.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.35)
value
}


#########
# intervals endpoints (assume the last one is 40)
v.40<-cut.points<-c(0,8,10,11,14,40)

Len.40<-leng(v.40)  # intervals lengths

standard.MAE.int.40.abalone<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V9.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.40)
value
}


#

################################################################################
####### preparing for k-fold cross-validation with k=10
#######

N=dim(abalone)[1]
n=round(N/10)

set.seed(12345)
fold<-sample(c(1:10),N,replace=TRUE)
table(fold)

training<-list()
test<-list()
sub.train<-list()

for (i in 1:10)
{test[[i]]<-abalone[which(fold==i),]
training[[i]]<-abalone[-which(fold==i),]}

# for (i in 1:10)
# {set.seed(12345)
#   random.sampl<-sample(which(fold!=i),2000,replace=FALSE)
#   sub.train[[i]]<-df[random.sampl,]}

################################################################################
################################################################################
########## e1071::tune.knn 
########## Classifier: knn
########## Resampling method: cross-validation with k=3
########## 
########## hyper-parameter to tune: k for the knn
##########
################################################################################

library(e1071)   # for tune.knn
library(class)   # for knn

####

tuned.knn.e1071.Accuracy.abalone<-list()
tuned.knn.e1071.MAE.abalone<-list()
tuned.knn.e1071.MAE.int.20.abalone<-list()
tuned.knn.e1071.MAE.int.25.abalone<-list()
tuned.knn.e1071.MAE.int.30.abalone<-list()
tuned.knn.e1071.MAE.int.35.abalone<-list()
tuned.knn.e1071.MAE.int.40.abalone<-list()

######## By default: error.fun = Error rate = 1 - Accuracy, to minimize

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.Accuracy.abalone[[i]] <- e1071::tune.knn(x = training[[i]][ ,c(2:8,14:16)],  y = training[[i]][ ,13],  
                                                   k=1:20,
                                                   tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=NULL))  # error.fun = Error rate
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.abalone[[i]] <- e1071::tune.knn(x = training[[i]][ ,c(2:8,14:16)],  y = training[[i]][ ,13],  
                                                   k=1:20,
                                                   tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.ord.abalone)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.20.abalone[[i]] <- e1071::tune.knn(x = training[[i]][ ,c(2:8,14:16)],  y = training[[i]][ ,13],  
                                              k=1:20,
                                              tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.20.abalone)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.25.abalone[[i]] <- e1071::tune.knn(x = training[[i]][ ,c(2:8,14:16)],  y = training[[i]][ ,13],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.25.abalone)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.30.abalone[[i]] <- e1071::tune.knn(x = training[[i]][ ,c(2:8,14:16)],  y = training[[i]][ ,13],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.30.abalone)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.35.abalone[[i]] <- e1071::tune.knn(x = training[[i]][ ,c(2:8,14:16)],  y = training[[i]][ ,13],  
                                                             k=1:20,
                                                             tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.35.abalone)) 
  
  print(i)}


#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.40.abalone[[i]] <- e1071::tune.knn(x = training[[i]][ ,c(2:8,14:16)],  y = training[[i]][ ,13],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.40.abalone)) 
  
  print(i)}


#
#

pred.tuned.knn.e1071.Accuracy.abalone<-list()
pred.tuned.knn.e1071.MAE.abalone<-list()
pred.tuned.knn.e1071.MAE.int.20.abalone<-list()
pred.tuned.knn.e1071.MAE.int.25.abalone<-list()
pred.tuned.knn.e1071.MAE.int.30.abalone<-list()
pred.tuned.knn.e1071.MAE.int.35.abalone<-list()
pred.tuned.knn.e1071.MAE.int.40.abalone<-list()

for (i in 1:10)
{pred.tuned.knn.e1071.Accuracy.abalone[[i]]<-class::knn(train=training[[i]][, c(2:8,14:16)], test=test[[i]][,c(2:8,14:16)],
                                                cl=training[[i]][,13],
                                              k=tuned.knn.e1071.Accuracy.abalone[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.abalone[[i]]<-class::knn(train=training[[i]][, c(2:8,14:16)], test=test[[i]][,c(2:8,14:16)],
                                                cl=training[[i]][,13],
                                                k=tuned.knn.e1071.MAE.abalone[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.20.abalone[[i]]<-class::knn(train=training[[i]][, c(2:8,14:16)], test=test[[i]][,c(2:8,14:16)],
                                                cl=training[[i]][,13],
                                                k=tuned.knn.e1071.MAE.int.20.abalone[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.25.abalone[[i]]<-class::knn(train=training[[i]][, c(2:8,14:16)], test=test[[i]][,c(2:8,14:16)],
                                                          cl=training[[i]][,13],
                                                          k=tuned.knn.e1071.MAE.int.25.abalone[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.30.abalone[[i]]<-class::knn(train=training[[i]][, c(2:8,14:16)], test=test[[i]][,c(2:8,14:16)],
                                                          cl=training[[i]][,13],
                                                          k=tuned.knn.e1071.MAE.int.30.abalone[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#


for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.35.abalone[[i]]<-class::knn(train=training[[i]][, c(2:8,14:16)], test=test[[i]][,c(2:8,14:16)],
                                                          cl=training[[i]][,13],
                                                          k=tuned.knn.e1071.MAE.int.35.abalone[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#


#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.40.abalone[[i]]<-class::knn(train=training[[i]][, c(2:8,14:16)], test=test[[i]][,c(2:8,14:16)],
                                                          cl=training[[i]][,13],
                                                          k=tuned.knn.e1071.MAE.int.40.abalone[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#
#

#
##
#


Conf.mat.tuned.knn.e1071.Accuracy.abalone<-list()
Conf.mat.tuned.knn.e1071.MAE.abalone<-list()
Conf.mat.tuned.knn.e1071.MAE.int.20.abalone<-list()
Conf.mat.tuned.knn.e1071.MAE.int.25.abalone<-list()
Conf.mat.tuned.knn.e1071.MAE.int.30.abalone<-list()
Conf.mat.tuned.knn.e1071.MAE.int.35.abalone<-list()
Conf.mat.tuned.knn.e1071.MAE.int.40.abalone<-list()


for (i in 1:10)
{Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]]<-
  table(pred.tuned.knn.e1071.Accuracy.abalone[[i]],test[[i]][,13])

Conf.mat.tuned.knn.e1071.MAE.abalone[[i]]<-
  table(pred.tuned.knn.e1071.MAE.abalone[[i]],test[[i]][,13])

Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.20.abalone[[i]],test[[i]][,13])

Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.25.abalone[[i]],test[[i]][,13])

Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.30.abalone[[i]],test[[i]][,13])

Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.35.abalone[[i]],test[[i]][,13])

Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.40.abalone[[i]],test[[i]][,13])

print(i)
}


################################################################################
################################################################################
################## Accuracy, SMAE and SMAE.int
################################################################################
#
Accuracy.M.Accuracy.abalone<-vector()
Accuracy.M.MAE.abalone<-vector()
Accuracy.M.MAE.int.20.abalone<-vector() 
Accuracy.M.MAE.int.25.abalone<-vector() 
Accuracy.M.MAE.int.30.abalone<-vector() 
Accuracy.M.MAE.int.35.abalone<-vector() 
Accuracy.M.MAE.int.40.abalone<-vector() 
#
SMAE.M.Accuracy.abalone<-vector()
SMAE.M.MAE.abalone<-vector()
SMAE.M.MAE.int.20.abalone<-vector()
SMAE.M.MAE.int.25.abalone<-vector()
SMAE.M.MAE.int.30.abalone<-vector()
SMAE.M.MAE.int.35.abalone<-vector()
SMAE.M.MAE.int.40.abalone<-vector()
#
SMAE.int.20.M.Accuracy.abalone<-vector()
SMAE.int.20.M.MAE.abalone<-vector()
SMAE.int.20.M.MAE.int.20.abalone<-vector()
SMAE.int.20.M.MAE.int.25.abalone<-vector()
SMAE.int.20.M.MAE.int.30.abalone<-vector()
SMAE.int.20.M.MAE.int.35.abalone<-vector()
SMAE.int.20.M.MAE.int.40.abalone<-vector()
#
SMAE.int.25.M.Accuracy.abalone<-vector()
SMAE.int.25.M.MAE.abalone<-vector()
SMAE.int.25.M.MAE.int.20.abalone<-vector()
SMAE.int.25.M.MAE.int.25.abalone<-vector()
SMAE.int.25.M.MAE.int.30.abalone<-vector()
SMAE.int.25.M.MAE.int.35.abalone<-vector()
SMAE.int.25.M.MAE.int.40.abalone<-vector()
#
SMAE.int.30.M.Accuracy.abalone<-vector()
SMAE.int.30.M.MAE.abalone<-vector()
SMAE.int.30.M.MAE.int.20.abalone<-vector()
SMAE.int.30.M.MAE.int.25.abalone<-vector()
SMAE.int.30.M.MAE.int.30.abalone<-vector()
SMAE.int.30.M.MAE.int.35.abalone<-vector()
SMAE.int.30.M.MAE.int.40.abalone<-vector()
#
SMAE.int.35.M.Accuracy.abalone<-vector()
SMAE.int.35.M.MAE.abalone<-vector()
SMAE.int.35.M.MAE.int.20.abalone<-vector()
SMAE.int.35.M.MAE.int.25.abalone<-vector()
SMAE.int.35.M.MAE.int.30.abalone<-vector()
SMAE.int.35.M.MAE.int.35.abalone<-vector()
SMAE.int.35.M.MAE.int.40.abalone<-vector()
#
SMAE.int.40.M.Accuracy.abalone<-vector()
SMAE.int.40.M.MAE.abalone<-vector()
SMAE.int.40.M.MAE.int.20.abalone<-vector()
SMAE.int.40.M.MAE.int.25.abalone<-vector()
SMAE.int.40.M.MAE.int.30.abalone<-vector()
SMAE.int.40.M.MAE.int.35.abalone<-vector()
SMAE.int.40.M.MAE.int.40.abalone<-vector()


for (i in 1:10)
{
  Accuracy.M.Accuracy.abalone[i]<-sum(diag(Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]]))/sum(Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]])
  Accuracy.M.MAE.abalone[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.abalone[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.abalone[[i]])
  Accuracy.M.MAE.int.20.abalone[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]])
  Accuracy.M.MAE.int.25.abalone[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]])
  Accuracy.M.MAE.int.30.abalone[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]])
  Accuracy.M.MAE.int.35.abalone[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]])
  Accuracy.M.MAE.int.40.abalone[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]])
  ##
  SMAE.M.Accuracy.abalone[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]],levels.V9.ordinal.encod))
  SMAE.M.MAE.abalone[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.abalone[[i]],levels.V9.ordinal.encod))
  SMAE.M.MAE.int.20.abalone[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]],levels.V9.ordinal.encod))
  SMAE.M.MAE.int.25.abalone[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]],levels.V9.ordinal.encod))
  SMAE.M.MAE.int.30.abalone[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]],levels.V9.ordinal.encod))
  SMAE.M.MAE.int.35.abalone[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]],levels.V9.ordinal.encod))
  SMAE.M.MAE.int.40.abalone[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]],levels.V9.ordinal.encod))
  ##
  SMAE.int.20.M.Accuracy.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]],levels.V9.ordinal.encod),Len.20)
  SMAE.int.20.M.MAE.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.abalone[[i]],levels.V9.ordinal.encod),Len.20)
  SMAE.int.20.M.MAE.int.20.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]],levels.V9.ordinal.encod),Len.20)
  SMAE.int.20.M.MAE.int.25.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]],levels.V9.ordinal.encod),Len.20)
  SMAE.int.20.M.MAE.int.30.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]],levels.V9.ordinal.encod),Len.20)
  SMAE.int.20.M.MAE.int.35.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]],levels.V9.ordinal.encod),Len.20)
  SMAE.int.20.M.MAE.int.40.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]],levels.V9.ordinal.encod),Len.20)
  ##
  SMAE.int.25.M.Accuracy.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]],levels.V9.ordinal.encod),Len.25)
  SMAE.int.25.M.MAE.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.abalone[[i]],levels.V9.ordinal.encod),Len.25)
  SMAE.int.25.M.MAE.int.20.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]],levels.V9.ordinal.encod),Len.25)
  SMAE.int.25.M.MAE.int.25.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]],levels.V9.ordinal.encod),Len.25)
  SMAE.int.25.M.MAE.int.30.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]],levels.V9.ordinal.encod),Len.25)
  SMAE.int.25.M.MAE.int.35.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]],levels.V9.ordinal.encod),Len.25)
  SMAE.int.25.M.MAE.int.40.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]],levels.V9.ordinal.encod),Len.25)
  ##
  SMAE.int.30.M.Accuracy.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]],levels.V9.ordinal.encod),Len.30)
  SMAE.int.30.M.MAE.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.abalone[[i]],levels.V9.ordinal.encod),Len.30)
  SMAE.int.30.M.MAE.int.20.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]],levels.V9.ordinal.encod),Len.30)
  SMAE.int.30.M.MAE.int.25.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]],levels.V9.ordinal.encod),Len.30)
  SMAE.int.30.M.MAE.int.30.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]],levels.V9.ordinal.encod),Len.30)
  SMAE.int.30.M.MAE.int.35.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]],levels.V9.ordinal.encod),Len.30)
  SMAE.int.30.M.MAE.int.40.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]],levels.V9.ordinal.encod),Len.30)
  ##
  SMAE.int.35.M.Accuracy.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]],levels.V9.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.abalone[[i]],levels.V9.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.20.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]],levels.V9.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.25.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]],levels.V9.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.30.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]],levels.V9.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.35.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]],levels.V9.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.40.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]],levels.V9.ordinal.encod),Len.35)
  ##
  SMAE.int.40.M.Accuracy.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.abalone[[i]],levels.V9.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.abalone[[i]],levels.V9.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.20.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.20.abalone[[i]],levels.V9.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.25.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.25.abalone[[i]],levels.V9.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.30.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.30.abalone[[i]],levels.V9.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.35.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.abalone[[i]],levels.V9.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.40.abalone[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.abalone[[i]],levels.V9.ordinal.encod),Len.40)
  }


###############################
###############################
###   SMAE.int

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot( 
    SMAE.int.20.M.MAE.int.20.abalone,
    SMAE.int.25.M.MAE.int.25.abalone,
    SMAE.int.30.M.MAE.int.30.abalone,
    SMAE.int.35.M.MAE.int.35.abalone,
    SMAE.int.40.M.MAE.int.40.abalone,
    main=paste(" "), 
    xlab="length assigned to the rigthmost interval", ylab=" ",names=c("6", "11", "16",
                                                                       "21", "26"),
    col=c2,medcol=c3)
  
}

#
#
#

Group.five<-c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10))


SMAE.int.all.abalone<-c(SMAE.int.20.M.MAE.int.20.abalone,
                        SMAE.int.25.M.MAE.int.25.abalone,
                        SMAE.int.30.M.MAE.int.30.abalone,
                        SMAE.int.40.M.MAE.int.35.abalone,
                        SMAE.int.50.M.MAE.int.40.abalone)

p1<-shapiro.test(SMAE.int.20.M.MAE.int.20.abalone)$p.value
p2<-shapiro.test(SMAE.int.25.M.MAE.int.25.abalone)$p.value
p3<-shapiro.test(SMAE.int.30.M.MAE.int.30.abalone)$p.value
p4<-shapiro.test(SMAE.int.35.M.MAE.int.35.abalone)$p.value
p5<-shapiro.test(SMAE.int.40.M.MAE.int.40.abalone)$p.value

if (p1>=0.05 & p2>=0.05 & p3>=0.05 & p4>=0.05 & p5>=0.05)
{
  test.SMAE.int.less.abalone<-pairwise.t.test(SMAE.int.all.abalone,Group.five,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater.abalone<-pairwise.t.test(SMAE.int.all.abalone,Group.five,paired=TRUE,alternative="greater", p.adjust.methods="holm")
} else {
  test.SMAE.int.less.abalone<-pairwise.wilcox.test(SMAE.int.all.abalone,Group.five,paired=TRUE,alternative="less", p.adjust.methods="holm")
  test.SMAE.int.greater.abalone<-pairwise.wilcox.test(SMAE.int.all.abalone,Group.five,paired=TRUE,alternative="greater", p.adjust.methods="holm")
}


test.SMAE.int.less.abalone
test.SMAE.int.greater.abalone

#
#




p1<-shapiro.test(SMAE.int.20.M.Accuracy.abalone)$p.value
p2<-shapiro.test(SMAE.int.20.M.MAE.abalone)$p.value
p3<-shapiro.test(SMAE.int.20.M.MAE.int.20.abalone)$p.value
p4<-shapiro.test(SMAE.int.20.M.MAE.int.25.abalone)$p.value
p5<-shapiro.test(SMAE.int.20.M.MAE.int.30.abalone)$p.value
p6<-shapiro.test(SMAE.int.20.M.MAE.int.35.abalone)$p.value
p7<-shapiro.test(SMAE.int.20.M.MAE.int.40.abalone)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.20.M.MAE.int.20.abalone,SMAE.int.20.M.Accuracy.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.20.M.MAE.int.20.abalone,SMAE.int.20.M.MAE.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.20.M.MAE.int.20.abalone,SMAE.int.20.M.MAE.int.25.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.20.M.MAE.int.20.abalone,SMAE.int.20.M.MAE.int.30.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.20.M.MAE.int.20.abalone,SMAE.int.20.M.MAE.int.35.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.20.M.MAE.int.20.abalone,SMAE.int.20.M.MAE.int.40.abalone, paired=TRUE, alternative="less")$p.value

p1<-shapiro.test(SMAE.int.25.M.Accuracy.abalone)$p.value
p2<-shapiro.test(SMAE.int.25.M.MAE.abalone)$p.value
p3<-shapiro.test(SMAE.int.25.M.MAE.int.20.abalone)$p.value
p4<-shapiro.test(SMAE.int.25.M.MAE.int.25.abalone)$p.value
p5<-shapiro.test(SMAE.int.25.M.MAE.int.30.abalone)$p.value
p6<-shapiro.test(SMAE.int.25.M.MAE.int.35.abalone)$p.value
p7<-shapiro.test(SMAE.int.25.M.MAE.int.40.abalone)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.25.M.MAE.int.25.abalone,SMAE.int.25.M.Accuracy.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.25.M.MAE.int.25.abalone,SMAE.int.25.M.MAE.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.25.M.MAE.int.25.abalone,SMAE.int.25.M.MAE.int.20.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.25.M.MAE.int.25.abalone,SMAE.int.25.M.MAE.int.30.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.25.M.MAE.int.25.abalone,SMAE.int.25.M.MAE.int.35.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.25.M.MAE.int.25.abalone,SMAE.int.25.M.MAE.int.40.abalone, paired=TRUE, alternative="less")$p.value


p1<-shapiro.test(SMAE.int.30.M.Accuracy.abalone)$p.value
p2<-shapiro.test(SMAE.int.30.M.MAE.abalone)$p.value
p3<-shapiro.test(SMAE.int.30.M.MAE.int.20.abalone)$p.value
p4<-shapiro.test(SMAE.int.30.M.MAE.int.25.abalone)$p.value
p5<-shapiro.test(SMAE.int.30.M.MAE.int.30.abalone)$p.value
p6<-shapiro.test(SMAE.int.30.M.MAE.int.35.abalone)$p.value
p7<-shapiro.test(SMAE.int.30.M.MAE.int.40.abalone)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.Accuracy.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.int.20.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.int.25.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.int.35.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.int.40.abalone, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.Accuracy.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.int.20.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.int.25.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.int.35.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.30.M.MAE.int.30.abalone,SMAE.int.30.M.MAE.int.40.abalone, paired=TRUE, alternative="greater")$p.value

p1<-shapiro.test(SMAE.int.35.M.Accuracy.abalone)$p.value
p2<-shapiro.test(SMAE.int.35.M.MAE.abalone)$p.value
p3<-shapiro.test(SMAE.int.35.M.MAE.int.20.abalone)$p.value
p4<-shapiro.test(SMAE.int.35.M.MAE.int.25.abalone)$p.value
p5<-shapiro.test(SMAE.int.35.M.MAE.int.30.abalone)$p.value
p6<-shapiro.test(SMAE.int.35.M.MAE.int.35.abalone)$p.value
p7<-shapiro.test(SMAE.int.35.M.MAE.int.40.abalone)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.Accuracy.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.int.20.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.int.25.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.int.30.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.int.40.abalone, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.Accuracy.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.int.20.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.int.25.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.int.30.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.35.M.MAE.int.35.abalone,SMAE.int.35.M.MAE.int.40.abalone, paired=TRUE, alternative="greater")$p.value


p1<-shapiro.test(SMAE.int.40.M.Accuracy.abalone)$p.value
p2<-shapiro.test(SMAE.int.40.M.MAE.abalone)$p.value
p3<-shapiro.test(SMAE.int.40.M.MAE.int.20.abalone)$p.value
p4<-shapiro.test(SMAE.int.40.M.MAE.int.25.abalone)$p.value
p5<-shapiro.test(SMAE.int.40.M.MAE.int.30.abalone)$p.value
p6<-shapiro.test(SMAE.int.40.M.MAE.int.35.abalone)$p.value
p7<-shapiro.test(SMAE.int.40.M.MAE.int.40.abalone)$p.value
p1;p2;p3;p4;p5;p6;p7

t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.Accuracy.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.int.20.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.int.25.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.int.30.abalone, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.int.35.abalone, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.Accuracy.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.int.20.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.int.25.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.int.30.abalone, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.abalone,SMAE.int.40.M.MAE.int.35.abalone, paired=TRUE, alternative="greater")$p.value

