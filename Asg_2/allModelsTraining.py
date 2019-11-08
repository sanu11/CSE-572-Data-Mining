import PCAonFeatureMatrix as pcafeature
import randomforest as rf
import deepLearningModel as nn_model

train_X,train_Y,test_X, test_Y = pcafeature.get_feature_matrix_Final('Data/concatenatedData.csv')

#1.Random Forest
#accuracy, precision, recall, F1score calculation for Random Forest
#accuracy, precision, recall, F1score =
rf.random_forest_train(train_X,train_Y)
rf.random_forest_test(test_X, test_Y)

#accuracy, precision, recall, F1score calculation for Neural Network
nn_model.train_neural_network(train_X,train_Y)
nn_model.test_neural_network(test_X, test_Y)

#accuracy, precision, recall, F1score calculation for Decision Tree -- ENTER HERE

#accuracy, precision, recall, F1score calculation for SVM -- ENTER HERE

#random forest test for one sample of meal or no meal : not tested yet!
def model1_test_one_sample(test_sample):
    #test_data = pcafeature.get_reduced_test_data(test_sample)
    output = rf.rf_test_one_sample(test_sample)
    print(output)
    return output

#model1_test_one_sample(test_X[0])
