import PCAonFeatureMatrix as pcafeature
import randomforest as rf
import deepLearningModel as nn_model
import svm
import decisiontree as dt

# 
test_X, test_Y,train_X,train_Y = pcafeature.get_feature_matrix_Final('Data/concatenatedData.csv')
<<<<<<< HEAD
#1.Random Forest

=======
>>>>>>> d7c5221fc5e365f34a6b61e63a316a5b421943e5

nn_model.train_neural_network(train_X,train_Y)
nn_model.test_neural_network(test_X, test_Y)

<<<<<<< HEAD

rf.random_forest_train(train_X,train_Y)
rf.random_forest_test(test_X, test_Y)



=======
rf.random_forest_train(train_X,train_Y)
rf.random_forest_test(test_X, test_Y)

>>>>>>> d7c5221fc5e365f34a6b61e63a316a5b421943e5
dt.decisiontrain(train_X,train_Y)
dt.decisiontest(test_X, test_Y)

svm.svm_train(train_X,train_Y)
svm.svm_test(test_X,test_Y)



#random forest test for one sample of meal or no meal : not tested yet!
def model1_test_one_sample(test_sample):
    #test_data = pcafeature.get_reduced_test_data(test_sample)
    output = rf.rf_test_one_sample(test_sample)
    #dt.dt_test_one(test_sample)
    print(output)
    return output

