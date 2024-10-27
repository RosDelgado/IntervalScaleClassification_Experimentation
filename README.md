# Performance metrics for ordinal and interval scale classification
In the field of supervised machine learning, the precise evaluation of classification models stands as a fundamental pursuit. This necessitates the utilization of robust performance metrics. 
This repository focuses on the real-world application of the interval-scale measures introduced in the paper

"Adapting Ordinal Performance Metrics for Interval Scale: Length Matters" by G. Binotto and R. Delgado (preprint, 2024)

for evaluation and as error functions for hyper-parameters tuning in classification with the dataset
https://www.kaggle.com/datasets/frabbisw/facial-age
(Section 5). 

It complements the content in https://github.com/giuliabinotto/ IntervalScaleClassification, which correspond to Section 4 fot the same paper, where scripts facilitate the computation of two ordinal metrics, Mean Absolute Error (MAE) and Total Cost (TC), alongside their interval scale counterparts introduced in the paper, with a specific section designed to address scenarios in which the rightmost interval is unbounded.

# Description
## From_png_to_dataframe.R 
This script allows to load face files (.png) obtained from https://www.kaggle.com/datasets/frabbisw/facial-age
and transform them into a dataframe, with 9,673 rows corresponding to face pictures, and 32x32+1=1025 columns, 
the last one being "age", while the others are the features V1,..., V1024. The dataframe is save as "faces.grey.32.Rda".

## train_caret_rf.R 
This script loads "faces.grey.32.Rda" and develops the experimental phase explained in Section 5 of the paper, corresponding 
use the interval-scale metrics to tuning hyper-parameter mtry for random forest using the caret library. 

## tune_control_e1071_knn.R 
This script is similar to the previous one, but corresponds to tuning hyper-parameter k for k-nearest neighbors (knn) using e1071 library.

# Requirements
## General R libraries
The following libraries are needed: 

magick, stringr, mdatools, png and utils (used by "From_png_to_dataframe.R")

arules (used by "train_caret_rf.R" and "tune_control_e1071.R")

caret (used by "train_caret_rf.R")

e1071 and class (used by "tune_control_e1071.R")

## Specific R scripts

mat_square.R (introduced here): converts any matrix in a square matrix with desired row/column labels, by adding zeros if needed.

From https://github.com/giuliabinotto/ IntervalScaleClassification

MAE.R: computes MAE and normalized SMAE metrics.

MAEintervals.R: computes MAE.int and SMAE.int metrics. 

# Authors
Giulia Binotto & Rosario Delgado (Universitat Autònoma de Barcelona, Spain, 2024).
