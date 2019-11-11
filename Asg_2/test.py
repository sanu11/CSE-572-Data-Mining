import sys
from numpy import genfromtxt
import numpy as np
import csv

import preprocessingGlucose as pg
import PCAonFeatureMatrix as pcafeature
import FeatureExtractionFFT as feature

import randomforest as rf
import deepLearningModel as nn_model
import decisiontree as dt
import svm



path = sys.argv[1]
pg.preprocess(path)
preprocessed_file = path+'_updated.csv'

# get preprocessed data
test_data = genfromtxt(preprocessed_file, delimiter=',')
n = len(test_data)

# test_data = np.delete(test_data,30,1)

# print("length of test data ",n, len(test_data[0]))


# get feature matrix
test_data = feature.feature_matrix_for_pca(preprocessed_file)

# remove labels column

print("test data length after feature matrix ",len(test_data))
test_data.drop(test_data.columns[1], axis=1,inplace=True)


# get pca data
test_X = pcafeature.get_reduced_test_data(test_data)

print("test data length after PCA",len(test_X))

# test_Y=[]

# with open(path) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         n = len(row)
#         test_Y.append(int(row[n-1]))

#  here, there is mismatch for test_X and test_Y rows

# print(test_Y,test_X)


#predict SVM
print("Predicitng result using Neural Network\n")
result= nn_model.nn_test_one_sample(test_X)
print(result)

print("Predicitng result using Random Forest\n")
result= rf.rf_test_one_sample(test_X)
print(result)

print("Predicitng result using SVM\n")
result= svm.svm_test_one_sample(test_X)
print(result)

print("Predicitng result using Decision Tree\n")
result= dt.dt_test_one(test_X)
print(result)
