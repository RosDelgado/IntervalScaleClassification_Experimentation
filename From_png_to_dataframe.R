
#######################################
#######################################
###
### Script to load face files (.png)
### and transform in a dataframe, with 
### faces by row, with 32*32+1=1025 columns, 
### the last one being "age", while the others
### are V1,..., V1024.
###
### Dataframe is save as "faces.grey.32.Rda", 
###             also as "faces.grey.32.csv"
###
### At Dades_FACES > face_age > Grey_pictures
###
######################################
######################################


# install.packages
library(magick)
library(stringr)
library(mdatools)
library(png)
library(utils)

##

folds.num<-c(1:93,95,96,99,100,101,110)  # = ages

Folder<-vector()
for (i in 1:9)
{
Folder[i] <- paste("~/Desktop/NEW_INVESTIGATION/ARTICULOS_Giulia/Metrics_ordinal_classification_ENVIADO_ML_2024_05_16/Machine_Learning_1a_REVISION/Dades_FACES/face_age/00",
                folds.num[i],sep="")
}

for (i in 10:96)
{
  Folder[i] <- paste("~/Desktop/NEW_INVESTIGATION/ARTICULOS_Giulia/Metrics_ordinal_classification_ENVIADO_ML_2024_05_16/Machine_Learning_1a_REVISION/Dades_FACES/face_age/0",
                     folds.num[i],sep="")
}


for (i in 97:99)
{
  Folder[i] <- paste("~/Desktop/NEW_INVESTIGATION/ARTICULOS_Giulia/Metrics_ordinal_classification_ENVIADO_ML_2024_05_16/Machine_Learning_1a_REVISION/Dades_FACES/face_age/",
                     folds.num[i],sep="")
}

##



file.png<-list()
for (i in 1:99)
{
file.png[[i]] <- list.files(Folder[i], pattern ="*.png", ignore.case = TRUE, full.names = TRUE)
}

##

picture.number<-list()

for (i in 1:99)
{picture.number[[i]]<-vector()
for (j in 1:length(file.png[[i]]))
{
  picture.number[[i]][j]<-stringr::str_sub(file.png[[i]][j], 
                                           start=c(166),
                                           end=-c(5)) 
}
}

###########
# read images .png with magick package

imgs<-list()
for (i in 1:99)
{imgs[[i]]<-magick::image_read(file.png[[i]], "png")
}


# # reduce to 64 x 64 pixels
# imgs.64<-list()
# for (i in 1:99)
# {
# imgs.64[[i]]<-magick::image_scale(imgs[[i]],"64x64!")
# }


# reduce to 32 x 32 pixels  
imgs.32<-list()
for (i in 1:99)
{
  imgs.32[[i]]<-magick::image_scale(imgs[[i]],"32x32!")
}

# set working directory at: 
# "~/Desktop/NEW_INVESTIGATION/ARTICULOS_Giulia/
# Metrics_ordinal_classification_ENVIADO_ML_2024_05_16/Machine_Learning_1a_REVISION/
# Dades_FACES/face_age/Grey_pictures" 
# and save the reduced images as .png

for (i in 1:99)
{
for (j in 1:length(imgs[[i]]))    
{
magick::image_write(imgs.32[[i]][j], paste("reduced_32x32_",picture.number[[i]][j],".png",sep=""))
}
}

# "reduced_number" are images reduced at 64 x 64 pixels


# Reading again as .png with png package
faces.small.32<-list()
for (i in 1:99)
{faces.small.32[[i]]<-list()
 for (j in 1:length(imgs[[i]]))
{
faces.small.32[[i]][[j]] <- png::readPNG(paste("reduced_32x32_",picture.number[[i]][j],".png",sep=""))
 }
}


### Be careful! Grey images have no RGB scales and then they provoke errors!
### We delete them (very few)
for (i in 1:99)
{
for (j in 0:(length(imgs[[i]])-1))
{
if (length(dim(faces.small.32[[i]][[length(imgs[[i]])-j]]))==2)  # grey faces have no RGB scales 
{faces.small.32[[i]][[length(imgs[[i]])-j]]<-NULL
picture.number[[i]]<-picture.number[[i]][-(length(imgs[[i]])-j)]}
}
}



# convert image to a data matrix with mdatools package
d.faces.32<-list()
for (i in 1:99)
{d.faces.32[[i]]<-list()
for (j in 1:length(faces.small.32[[i]]))
{
  d.faces.32[[i]][[j]] <- mdatools::mda.im2data(faces.small.32[[i]][[j]])
}
}


################
# show data values with mdatools, and also images
mdatools::mda.show(d.faces.32[[37]][[1]], 10)

par(mfrow = c(2, 2))
mdatools::imshow(d.faces.32[[37]][[1]], 1)
mdatools::imshow(d.faces.32[[37]][[1]], 2)
mdatools::imshow(d.faces.32[[37]][[1]], 3)
mdatools::imshow(d.faces.32[[37]][[1]], 1:3)
###########


# convert data matrix of any image to grey scale by linear combination of scales
# red (R), green (G) and blue (B)
grey.32<-list()
for (i in 1:99)
{grey.32[[i]]<-list()
for (j in 1:length(d.faces.32[[i]]))
{
  grey.32[[i]][[j]] <- 0.299*d.faces.32[[i]][[j]][,1]+
                    0.587*d.faces.32[[i]][[j]][,2]+
                    0.114*d.faces.32[[i]][[j]][,3] # from RGB to grey scale
}
}


# conver the data corresponding to faces in grey to a dataframe, "faces.grey.32", with
# rownames equal to the number of each picture in the original png file names
faces.grey.32<-list()
for (i in 1:99)
{faces.grey.32[[i]]<-as.data.frame(t(as.data.frame(grey.32[[i]])))
 rownames(faces.grey.32[[i]])<-picture.number[[i]]
}

str(faces.grey.32[[1]])  # 32*32=1024 variables (variable = pixel), 1105 images 1 year
str(faces.grey.32[[99]])  # 32*32=1024 variables (variable = pixel), 2 images 110 year


# construct the variable "age", which coincides with the number of the folder where
# the original face images are
age<-list()
for (i in 1:99)
{
age[[i]]<-rep(folds.num[i],length(picture.number[[i]])) 
}



# add "age" variable as last column to the dataframe "faces.grey"
for (i in 1:99)
{
faces.grey.32[[i]]<-as.data.frame(cbind(faces.grey.32[[i]],age[[i]]))
colnames(faces.grey.32[[i]])[32*32+1]<-"age"
}


str(faces.grey.32[[1]]) # 1025 variables: features "V1" to "V1024" + "age" (in year)

View(faces.grey.32[[1]][1:10,1000:1025])

###################################################
#### FINAL DATAFRAME. 9673 rows (=faces), 32*32 = 1024 features V1,... V1024 + "age" (1 to 110)
##################################################
db.faces.grey.32<-faces.grey.32[[1]]
for (i in 2:99)
{db.faces.grey.32<-rbind(db.faces.grey.32,faces.grey.32[[i]])}

str(db.faces.grey.32)
table(db.faces.grey.32[[1025]])


save(db.faces.grey.32,file="faces.grey.32.Rda")
utils::write.csv(db.faces.grey.32,file="faces.grey.32.csv")

