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
	nn_labels=[]
	print("Predicitng result using Neural Network: ")

	for i in range(len(test_X)) : 
		test_frame = test_X.iloc[i,:].to_frame().transpose()
		result= nn_model.nn_test_one_sample(test_frame)
		nn_labels.append(result)

	print(nn_labels)
	return nn_labels


def test_randomForest(test_X):
	rf_labels=[]
	print("Predicitng result using Random Forest: ")

	for i in range(len(test_X)) : 
		test_frame = test_X.iloc[i,:].to_frame().transpose()
		result= rf.rf_test_one_sample(test_X)
		rf_labels.append(result[0])

	print(rf_labels)
	return rf_labels


def test_svm(test_X):
	svm_labels=[]
	print("Predicitng result using SVM: ")

	for i in range(len(test_X)) : 
		test_frame = test_X.iloc[i,:].to_frame().transpose()
		result= svm.svm_test_one_sample(test_frame)
		svm_labels.append(result[0])

	print(svm_labels)	
	return svm_labels


def test_decisionTree(test_X):
	print("Predicitng result using Decision Tree: ")
	dt_labels=[]
	for i in range(len(test_X)) : 
		test_frame = test_X.iloc[i,:].to_frame().transpose()
		result= dt.dt_test_one(test_X)
		dt_labels.append(result[0])

	print(dt_labels)
	return dt_labels


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








