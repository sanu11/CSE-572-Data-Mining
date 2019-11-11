import sys
from numpy import genfromtxt
import numpy as np
import csv

import preprocessingGlucose as pg
import PCAonFeatureMatrix as pcafeature
import FeatureExtractionFFTForTest as feature

import randomforest as rf
import deepLearningModel as nn_model
import decisiontree as dt
import svm

def preprocess():

	path = sys.argv[1]
	pg.preprocess(path)
	preprocessed_file = path+'_updated.csv'

	# get preprocessed data
	test_data = genfromtxt(preprocessed_file, delimiter=',')
	n = len(test_data)

	# get feature matrix
	test_data = feature.feature_matrix_for_pca(preprocessed_file)

	# remove labels column
	print("test data length after feature matrix ",len(test_data))
	test_data.drop(test_data.columns[1], axis=1,inplace=True)

	# get pca data
	test_X = pcafeature.get_reduced_test_data(test_data)
	print("test data length after PCA",len(test_X))
	return test_X


def test_neuralNetwork(test_X):
	print("Predicitng result using Neural Network: ")
	result= nn_model.nn_test_one_sample(test_X)
	print(result)
	return result


def test_randomForest(test_X):
	print("Predicitng result using Random Forest: ")
	result= rf.rf_test_one_sample(test_X)
	print(result)
	return result


def test_svm(test_X):
	print("Predicitng result using SVM: ")
	result= svm.svm_test_one_sample(test_X)
	print(result)	
	return result


def test_decisionTree(test_X):
	print("Predicitng result using Decision Tree: ")
	result= dt.dt_test_one(test_X)
	print(result)
	return result


# test_Y=[]

# with open(path) as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         n = len(row)
#         test_Y.append(int(row[n-1]))

#  here, there is mismatch for test_X and test_Y rows

# print(test_Y,test_X)

test_X = preprocess()
test_neuralNetwork(test_X)
test_randomForest(test_X)
test_svm(test_X)
test_decisionTree(test_X)







