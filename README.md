# Performance metrics for ordinal and interval scale classification
In the field of supervised machine learning, the precise evaluation of classification models stands as a fundamental pursuit. This necessitates the utilization of robust performance metrics. 
This repository focuses on the real-world application of the interval-scale measures introduced in the paper

"Adapting Ordinal Performance Metrics for Interval Scale: Length Matters" by G. Binotto and R. Delgado (preprint, 2024)

for evaluation and as error functions for hyper-parameters tuning in classification with the dataset
https://www.kaggle.com/datasets/frabbisw/facial-age
(Section 5). 

It complements the content in https://github.com/giuliabinotto/ IntervalScaleClassification, which correspond to Section 4 fot the same paper, where scripts facilitate the computation of two ordinal metrics, Mean Absolute Error (MAE) and Total Cost (TC), alongside their interval scale counterparts introduced in the paper, with a specific section designed to address scenarios in which the rightmost interval is unbounded.

# Requirements
The following libraries are needed: 

From_png_to_dataframe.R uses: magick, stringr, mdatools, png, utils.

train_caret_rf.R uses: arules (for discretize function), and caret. 

tune_control_e1071_knn.R uses: arules (for discretize function), e1071 and class.

The following scripts are needed: 

mat_square.R: converts any matrix in a square matrix with desired row/column labels, by adding zeros if needed.

From https://github.com/giuliabinotto/ IntervalScaleClassification

MAE.R: computes MAE and normalized SMAE metrics.

MAEintervals.R: computes MAE.int and SMAE.int metrics. 

# Description
The From_png_to_dataframe.R script allows to load face files (.png) and transform then into a dataframe, with face pictures by row, with 32x32+1=1025 columns, the last one being "age", while the others are V1,..., V1024.The dataframe is save as "faces.grey.32.Rda".

The train_caret_rf.R script loads "faces.grey.32.Rda" and develops the experimental phase by tuning hyper-parameter mtry for random forest.

The tune_control_e1071_knn.R the same, with hyper-parameter k for k-nearest neighbors (knn).

# Authors
Giulia Binotto & Rosario Delgado (Universitat Autònoma de Barcelona, Spain, 2024).
