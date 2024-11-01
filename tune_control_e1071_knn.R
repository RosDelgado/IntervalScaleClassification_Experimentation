###########################################
###########################################
#
#  EXPERIMENTAL PHASE (Section 5)
#
# Tuning knn with e1071
#
# Data set: face_age
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

library(arules)   # for "discretize" function
library(doParallel)
registerDoParallel(cores=6)



load("faces.grey.32.Rda")  # load dataframe "db.faces.grey.32"
df<-db.faces.grey.32
str(df)


# #####  variable "age": binning to "age.bin", with 5 intervals

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

df$age.bin.num<-as.numeric(df$age.bin)
table(df$age.bin)

df$age.bin.num.factor<-as.factor(df$age.bin.num)   # classes in number but factor type 


################################################################################
####### Error functions for tune.control argument, tune function
#######

standard.MAE.ord<-function(y,z)   # y = true, z = predicted
{Conf.mat<- mat.square(table(as.numeric(z),as.numeric(y)),levels.age.ordinal.encod)
  value<-SMAE(Conf.mat)
  value
}

#########

v.80<-c(0,2,10,15,35,60,80) # intervals endpoints (assume the last one is 80)

Len.80<-leng(v.80)  # intervals lengths

standard.MAE.int.80<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.age.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.80)
value
}


#########

v.90<-c(0,2,10,15,35,60,90) # intervals endpoints (assume the last one is 90)

Len.90<-leng(v.90)  # intervals lengths

standard.MAE.int.90<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.age.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.90)
value
}

#########

v.100<-c(0,2,10,15,35,60,100) # intervals endpoints (assume the last one is 100)

Len.100<-leng(v.100)  # intervals lengths

standard.MAE.int.100<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.age.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.100)
value
}

#########

v.110<-c(0,2,10,15,35,60,110) # intervals endpoints (assume the last one is 110)

Len.110<-leng(v.110)  # intervals lengths

standard.MAE.int.110<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.age.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.110)
value
}

#########

v.120<-c(0,2,10,15,35,60,120) # intervals endpoints (assume the last one is 120)

Len.120<-leng(v.120)  # intervals lengths

standard.MAE.int.120<-function(y,z)   # y = true, z = predicted
{Conf.mat<-mat.square(table(as.numeric(z),as.numeric(y)),levels.age.ordinal.encod)
value<-SMAE.int(Conf.mat,Len.120)
value
}



################################################################################
####### 
####### Preparing for k-fold cross-validation with k=10
#######

N=dim(df)[1]
n=round(N/10)

set.seed(12345)
fold<-sample(c(1:10),N,replace=TRUE)
table(fold)

training<-list()
test<-list()
sub.train<-list()

for (i in 1:10)
{test[[i]]<-df[which(fold==i),]
training[[i]]<-df[-which(fold==i),]}

# random sample of size 2000 from any training set:

for (i in 1:10)
{set.seed(12345)
  random.sampl<-sample(which(fold!=i),2000,replace=FALSE)
  sub.train[[i]]<-df[random.sampl,]}

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

tuned.knn.e1071.Accuracy<-list()
tuned.knn.e1071.MAE<-list()
tuned.knn.e1071.MAE.int.80<-list()
tuned.knn.e1071.MAE.int.90<-list()
tuned.knn.e1071.MAE.int.100<-list()
tuned.knn.e1071.MAE.int.110<-list()
tuned.knn.e1071.MAE.int.120<-list()

######## By default: error.fun = Error rate = 1 - Accuracy, to minimize

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.Accuracy[[i]] <- e1071::tune.knn(x = sub.train[[i]][ ,-c(1025:1028)],  y = sub.train[[i]][ ,1028],  
                                                   k=1:20,
                                                   tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=NULL))  # error.fun = Error rate
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE[[i]] <- e1071::tune.knn(x = sub.train[[i]][ ,-c(1025:1028)],  y = sub.train[[i]][ ,1028],  
                                                   k=1:20,
                                                   tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.ord)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.80[[i]] <- e1071::tune.knn(x = sub.train[[i]][ ,-c(1025:1028)],  y = sub.train[[i]][ ,1028],  
                                              k=1:20,
                                              tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.80)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.90[[i]] <- e1071::tune.knn(x = sub.train[[i]][ ,-c(1025:1028)],  y = sub.train[[i]][ ,1028],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.90)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.100[[i]] <- e1071::tune.knn(x = sub.train[[i]][ ,-c(1025:1028)],  y = sub.train[[i]][ ,1028],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.100)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.110[[i]] <- e1071::tune.knn(x = sub.train[[i]][ ,-c(1025:1028)],  y = sub.train[[i]][ ,1028],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.110)) 
  
  print(i)}

#

for (i in 1:10)
{ set.seed(12345)
  tuned.knn.e1071.MAE.int.120[[i]] <- e1071::tune.knn(x = sub.train[[i]][ ,-c(1025:1028)],  y = sub.train[[i]][ ,1028],  
                                                     k=1:20,
                                                     tunecontrol = tune.control(sampling = "cross", cross=3, error.fun=standard.MAE.int.120)) 
  
  print(i)}



#
#

pred.tuned.knn.e1071.Accuracy<-list()
pred.tuned.knn.e1071.MAE<-list()
pred.tuned.knn.e1071.MAE.int.80<-list()
pred.tuned.knn.e1071.MAE.int.90<-list()
pred.tuned.knn.e1071.MAE.int.100<-list()
pred.tuned.knn.e1071.MAE.int.110<-list()
pred.tuned.knn.e1071.MAE.int.120<-list()

for (i in 1:10)
{pred.tuned.knn.e1071.Accuracy[[i]]<-class::knn(train=sub.train[[i]][, -c(1025:1028)], test=test[[i]][,-c(1025:1028)],
                                                cl=sub.train[[i]][,1028],
                                              k=tuned.knn.e1071.Accuracy[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE[[i]]<-class::knn(train=sub.train[[i]][, -c(1025:1028)], test=test[[i]][,-c(1025:1028)],
                                                cl=sub.train[[i]][,1028],
                                                k=tuned.knn.e1071.MAE[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.80[[i]]<-class::knn(train=sub.train[[i]][, -c(1025:1028)], test=test[[i]][,-c(1025:1028)],
                                                cl=sub.train[[i]][,1028],
                                                k=tuned.knn.e1071.MAE.int.80[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.90[[i]]<-class::knn(train=sub.train[[i]][, -c(1025:1028)], test=test[[i]][,-c(1025:1028)],
                                                  cl=sub.train[[i]][,1028],
                                                  k=tuned.knn.e1071.MAE.int.90[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.100[[i]]<-class::knn(train=sub.train[[i]][, -c(1025:1028)], test=test[[i]][,-c(1025:1028)],
                                                  cl=sub.train[[i]][,1028],
                                                  k=tuned.knn.e1071.MAE.int.100[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.110[[i]]<-class::knn(train=sub.train[[i]][, -c(1025:1028)], test=test[[i]][,-c(1025:1028)],
                                                  cl=sub.train[[i]][,1028],
                                                  k=tuned.knn.e1071.MAE.int.110[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#

for (i in 1:10)
{pred.tuned.knn.e1071.MAE.int.120[[i]]<-class::knn(train=sub.train[[i]][, -c(1025:1028)], test=test[[i]][,-c(1025:1028)],
                                                  cl=sub.train[[i]][,1028],
                                                  k=tuned.knn.e1071.MAE.int.120[[i]]$best.parameters,prob=FALSE, use.all=TRUE)
print(i)
}

#
##
#


Conf.mat.tuned.knn.e1071.Accuracy<-list()
Conf.mat.tuned.knn.e1071.MAE<-list()
Conf.mat.tuned.knn.e1071.MAE.int.80<-list()
Conf.mat.tuned.knn.e1071.MAE.int.90<-list()
Conf.mat.tuned.knn.e1071.MAE.int.100<-list()
Conf.mat.tuned.knn.e1071.MAE.int.110<-list()
Conf.mat.tuned.knn.e1071.MAE.int.120<-list()


for (i in 1:10)
{Conf.mat.tuned.knn.e1071.Accuracy[[i]]<-
  table(pred.tuned.knn.e1071.Accuracy[[i]],test[[i]][,1028])

Conf.mat.tuned.knn.e1071.MAE[[i]]<-
  table(pred.tuned.knn.e1071.MAE[[i]],test[[i]][,1028])

Conf.mat.tuned.knn.e1071.MAE.int.80[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.80[[i]],test[[i]][,1028])

Conf.mat.tuned.knn.e1071.MAE.int.90[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.90[[i]],test[[i]][,1028])

Conf.mat.tuned.knn.e1071.MAE.int.100[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.100[[i]],test[[i]][,1028])

Conf.mat.tuned.knn.e1071.MAE.int.110[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.110[[i]],test[[i]][,1028])

Conf.mat.tuned.knn.e1071.MAE.int.120[[i]]<-
  table(pred.tuned.knn.e1071.MAE.int.120[[i]],test[[i]][,1028])

print(i)
}

Conf.mat.tuned.knn.e1071.Accuracy
Conf.mat.tuned.knn.e1071.MAE
Conf.mat.tuned.knn.e1071.MAE.int.80
Conf.mat.tuned.knn.e1071.MAE.int.90
Conf.mat.tuned.knn.e1071.MAE.int.100
Conf.mat.tuned.knn.e1071.MAE.int.110
Conf.mat.tuned.knn.e1071.MAE.int.120

################################################################################
################################################################################
################## SMAE.int computation, boxplots and tests
################################################################################
#
SMAE.int.80.M.MAE.int.80<-vector()
SMAE.int.90.M.MAE.int.90<-vector()
SMAE.int.100.M.MAE.int.100<-vector()
SMAE.int.110.M.MAE.int.110<-vector()
SMAE.int.120.M.MAE.int.120<-vector()

for (i in 1:10)
{
  SMAE.int.80.M.MAE.int.80[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.80[[i]],levels.age.ordinal.encod),Len.80)
  SMAE.int.90.M.MAE.int.90[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.90[[i]],levels.age.ordinal.encod),Len.90)
  SMAE.int.100.M.MAE.int.100[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.100[[i]],levels.age.ordinal.encod),Len.100)
  SMAE.int.110.M.MAE.int.110[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.110[[i]],levels.age.ordinal.encod),Len.110)
  SMAE.int.120.M.MAE.int.120[i]<-SMAE.int(mat.square(Conf.mat.tuned.knn.e1071.MAE.int.120[[i]],levels.age.ordinal.encod),Len.120)
}


mean(SMAE.int.80.M.MAE.int.80,na.rm=TRUE)
mean(SMAE.int.90.M.MAE.int.90,na.rm=TRUE)
mean(SMAE.int.100.M.MAE.int.100,na.rm=TRUE)
mean(SMAE.int.110.M.MAE.int.110,na.rm=TRUE)
mean(SMAE.int.120.M.MAE.int.120,na.rm=TRUE)


###############################
###############################
### BOXPLOTS and statistical tests for multiple comparisons
###

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

for (i in 1:length(seeds)){
  boxplot(SMAE.int.80.M.MAE.int.80,
          SMAE.int.90.M.MAE.int.90,
          SMAE.int.100.M.MAE.int.100,
          SMAE.int.110.M.MAE.int.110,
          SMAE.int.120.M.MAE.int.120,
          #main=paste(" "), 
          xlab="length assigned to the rightmost interval", ylab=" ",names=c("20","30","40","50","60"),
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

