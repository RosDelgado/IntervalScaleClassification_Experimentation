###########################################
###########################################
#
#  EXPERIMENTAL PHASE (Section 5)
#
# Tuning knn with e1071
#
# Data set: Parkinson
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
####### Error functions for tune.control argument, tune function
#######

standard.MAE.ord.parkinson<-function(y,z)   # y = true, z = predicted
{Conf.mat<- mat.square(table(as.numeric(z),as.numeric(y)),levels.V5.ordinal.encod)
value<-SMAE(Conf.mat)
value
}

#########
# intervals endpoints (assume the last one is 35)
v.35<-c(0,13,18,24,29,35)

Len.35<-leng(v.35)  # intervals lengths

standard.MAE.int.35.parkinson<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V5.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.35)
value
}
 
#########
# intervals endpoints (assume the last one is 40)
v.40<-cut.points<-c(0,13,18,24,29,40)

Len.40<-leng(v.40)  # intervals lengths

standard.MAE.int.40.parkinson<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V5.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.40)
value
}

#########
# intervals endpoints (assume the last one is 45)
v.45<-cut.points<-c(0,13,18,24,29,45)

Len.45<-leng(v.45)  # intervals lengths

standard.MAE.int.45.parkinson<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V5.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.45)
value
}

#########
# intervals endpoints (assume the last one is 50)
v.50<-cut.points<-c(0,13,18,24,29,50)

Len.50<-leng(v.50)  # intervals lengths

standard.MAE.int.50.parkinson<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V5.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.50)
value
}

#########
# intervals endpoints (assume the last one is 60)
v.60<-cut.points<-c(0,13,18,24,29,60)

Len.60<-leng(v.60)  # intervals lengths

standard.MAE.int.60.parkinson<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.V5.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.60)
value
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

tuned.knn.e1071.Accuracy.parkinson<-list()
tuned.knn.e1071.MAE.parkinson<-list()
tuned.knn.e1071.MAE.int.35.parkinson<-list()
tuned.knn.e1071.MAE.int.40.parkinson<-list()
tuned.knn.e1071.MAE.int.45.parkinson<-list()
tuned.knn.e1071.MAE.int.50.parkinson<-list()
tuned.knn.e1071.MAE.int.60.parkinson<-list()

######## By default: error.fun = Error rate = 1 - Accuracy, to minimize

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.Accuracy.parkinson[[i]] <- e1071::tune.knn(x = training[[i]][ ,features],  y = training[[i]][ ,25],  
                                                   k=1:20,
                                                   tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=NULL))  # error.fun = Error rate
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.parkinson[[i]] <- e1071::tune.knn(x = training[[i]][ ,features],  y = training[[i]][ ,25],  
                                                   k=1:20,
                                                   tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.ord.parkinson)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.35.parkinson[[i]] <- e1071::tune.knn(x = training[[i]][ ,features],  y = training[[i]][ ,25],  
                                              k=1:20,
                                              tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.35.parkinson)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.40.parkinson[[i]] <- e1071::tune.knn(x = training[[i]][ ,features],  y = training[[i]][ ,25],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.40.parkinson)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.45.parkinson[[i]] <- e1071::tune.knn(x = training[[i]][ ,features],  y = training[[i]][ ,25],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.45.parkinson)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.50.parkinson[[i]] <- e1071::tune.knn(x = training[[i]][ ,features],  y = training[[i]][ ,25],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.50.parkinson)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.60.parkinson[[i]] <- e1071::tune.knn(x = training[[i]][ ,features],  y = training[[i]][ ,25],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.60.parkinson)) 
  
  print(i)}


#
#

pred.tuned.knn.e1071.Accuracy.parkinson<-list()
pred.tuned.knn.e1071.MAE.parkinson<-list()
pred.tuned.knn.e1071.MAE.int.35.parkinson<-list()
pred.tuned.knn.e1071.MAE.int.40.parkinson<-list()
pred.tuned.knn.e1071.MAE.int.45.parkinson<-list()
pred.tuned.knn.e1071.MAE.int.50.parkinson<-list()
pred.tuned.knn.e1071.MAE.int.60.parkinson<-list()

for (i in 1:10)
{pred.tuned.knn.e1071.Accuracy.parkinson[[i]]<-class::knn(train=training[[i]][, features], test=test[[i]][,features],
                                                cl=training[[i]][,25],
                                              k=tuned.knn.e1071.Accuracy.parkinson[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.parkinson[[i]]<-class::knn(train=training[[i]][, features], test=test[[i]][,features],
                                                cl=training[[i]][,25],
                                                k=tuned.knn.e1071.MAE.parkinson[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.35.parkinson[[i]]<-class::knn(train=training[[i]][, features], test=test[[i]][,features],
                                                cl=training[[i]][,25],
                                                k=tuned.knn.e1071.MAE.int.35.parkinson[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.40.parkinson[[i]]<-class::knn(train=training[[i]][, features], test=test[[i]][,features],
                                                          cl=training[[i]][,25],
                                                          k=tuned.knn.e1071.MAE.int.40.parkinson[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.45.parkinson[[i]]<-class::knn(train=training[[i]][, features], test=test[[i]][,features],
                                                          cl=training[[i]][,25],
                                                          k=tuned.knn.e1071.MAE.int.45.parkinson[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#


#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.50.parkinson[[i]]<-class::knn(train=training[[i]][, features], test=test[[i]][,features],
                                                          cl=training[[i]][,25],
                                                          k=tuned.knn.e1071.MAE.int.50.parkinson[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.60.parkinson[[i]]<-class::knn(train=training[[i]][, features], test=test[[i]][,features],
                                                          cl=training[[i]][,25],
                                                          k=tuned.knn.e1071.MAE.int.60.parkinson[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

#

#
##
#


Conf.mat.tuned.knn.e1071.Accuracy.parkinson<-list()
Conf.mat.tuned.knn.e1071.MAE.parkinson<-list()
Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson<-list()
Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson<-list()
Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson<-list()
Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson<-list()
Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson<-list()


for (i in 1:10)
{Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]]<-
  table(pred.tuned.knn.e1071.Accuracy.parkinson[[i]],test[[i]][,25])

Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]]<-
  table(pred.tuned.knn.e1071.MAE.parkinson[[i]],test[[i]][,25])

Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.35.parkinson[[i]],test[[i]][,25])

Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.40.parkinson[[i]],test[[i]][,25])

Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.45.parkinson[[i]],test[[i]][,25])

Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.50.parkinson[[i]],test[[i]][,25])

Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.60.parkinson[[i]],test[[i]][,25])

print(i)
}


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
  Accuracy.M.Accuracy.parkinson[i]<-sum(diag(Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]]))/sum(Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]])
  Accuracy.M.MAE.parkinson[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]])
  Accuracy.M.MAE.int.35.parkinson[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]])
  Accuracy.M.MAE.int.40.parkinson[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]])
  Accuracy.M.MAE.int.45.parkinson[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]])
  Accuracy.M.MAE.int.50.parkinson[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]])
  Accuracy.M.MAE.int.60.parkinson[i]<-sum(diag(Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]]))/sum(Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]])
  ##
  SMAE.M.Accuracy.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.35.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.40.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.45.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.50.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod))
  SMAE.M.MAE.int.60.parkinson[i]<-SMAE(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod))
  ##
  SMAE.int.35.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  SMAE.int.35.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.35)
  ##
  SMAE.int.40.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  SMAE.int.40.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.40)
  ##
  SMAE.int.45.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  SMAE.int.45.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.45)
  ##
  SMAE.int.50.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  SMAE.int.50.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.50)
  ##
  SMAE.int.60.M.Accuracy.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.Accuracy.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.35.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.35.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.40.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.40.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.45.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.45.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.50.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.50.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
  SMAE.int.60.M.MAE.int.60.parkinson[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.60.parkinson[[i]],levels.V5.ordinal.encod),Len.60)
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
wilcox.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.45.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.50.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.60.parkinson, paired=TRUE, alternative="less")$p.value

t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.Accuracy.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.35.parkinson, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.40.M.MAE.int.40.parkinson,SMAE.int.40.M.MAE.int.45.parkinson, paired=TRUE, alternative="greater")$p.value
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

wilcox.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.Accuracy.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.35.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.40.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.50.parkinson, paired=TRUE, alternative="less")$p.value
wilcox.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.60.parkinson, paired=TRUE, alternative="less")$p.value

wilcox.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.Accuracy.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.35.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.40.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.50.parkinson, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.45.M.MAE.int.45.parkinson,SMAE.int.45.M.MAE.int.60.parkinson, paired=TRUE, alternative="greater")$p.value

# #

p1<-shapiro.test(SMAE.int.50.M.Accuracy.parkinson)$p.value
p2<-shapiro.test(SMAE.int.50.M.MAE.parkinson)$p.value
p3<-shapiro.test(SMAE.int.50.M.MAE.int.35.parkinson)$p.value
p4<-shapiro.test(SMAE.int.50.M.MAE.int.40.parkinson)$p.value
p5<-shapiro.test(SMAE.int.50.M.MAE.int.45.parkinson)$p.value
p6<-shapiro.test(SMAE.int.50.M.MAE.int.50.parkinson)$p.value
p7<-shapiro.test(SMAE.int.50.M.MAE.int.60.parkinson)$p.value
p1;p2;p3;p4;p5;p6;p7

wilcox.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.Accuracy.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.35.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.40.parkinson, paired=TRUE, alternative="less")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.45.parkinson, paired=TRUE, alternative="less")$p.value
wilcox.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.60.parkinson, paired=TRUE, alternative="less")$p.value

wilcox.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.Accuracy.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.35.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.40.parkinson, paired=TRUE, alternative="greater")$p.value
t.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.45.parkinson, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.50.M.MAE.int.50.parkinson,SMAE.int.50.M.MAE.int.60.parkinson, paired=TRUE, alternative="greater")$p.value

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


