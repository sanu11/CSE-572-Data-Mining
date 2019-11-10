import preprocessingGlucose as pg
import PCAonFeatureMatrix as pcafeature
import randomforest as rf
# import deepLearningModel as nn_model
import svm
import sys
from numpy import genfromtxt
import numpy as np
import FeatureExtractionFFT as feature
import csv

path = sys.argv[1]
pg.preprocess(path)
preprocessed_file = path+'_updated.csv'

# get preprocessed data
test_data = genfromtxt(preprocessed_file, delimiter=',')
n = len(test_data)

print("length of test data ",n, len(test_data[0]))

# for i in range(0,n):
# test_data = np.delete(test_data,30,1)

# print len(test_data)
# get feature matrix
test_data = feature.feature_matrix_for_pca(preprocessed_file)

print("test data length after feature matrix ",len(test_data))
test_data.drop(test_data.columns[1], axis=1,inplace=True)



# get pca data
test_X = 	pcafeature.get_reduced_test_data(test_data)
print("test data length after PCA",len(test_X))

test_Y=[]
with open(path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        n = len(row)
        test_Y.append(int(row[n-1]))

print(test_Y,test_X)

# Test all models
rf.random_forest_test(test_X, test_Y)
nn_model.test_neural_network(test_X,test_Y)
svm.svm_test(test_X,test_Y)

# 
