# Import the model we are using
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

import pickle
import numpy as np


def svm_train(train_features,train_labels):
    print("Starting SVM Training Model")
    # Instantiate model with 1000 decision trees
    # Using K-Fold to get Samples
    clf = SVC(kernel='linear', C=1)  
    clf.fit(train_features,train_labels)
    kfold_accuracies=[]
    kfold_accuracies = cross_val_score(clf, train_features, train_labels, cv=4)
    print("\n\n*********** SVM Output******************")
    print("Accuracies K Fold: ",kfold_accuracies)
    print("SVM Mean Accuracy(K fold): ", sum(kfold_accuracies)/len(kfold_accuracies))
    pickle.dump(clf,open('svm_model.sav','wb'))
    # Use the forest's predict method on the test data

def svm_test(test_features,test_labels):
    rf = pickle.load(open('svm_model.sav', 'rb'))

    predictions = rf.predict(test_features)

    print("**************Test Accuracy****************")

    print("SVM accuracy = ",accuracy_score(test_labels,predictions))
    print("SVM precision = ",precision_score(test_labels,predictions))
    print("SVM recall = ",recall_score(test_labels,predictions))
    print("SVM F1 = ",f1_score(test_labels,predictions))
    print("\n")


    # Calculate roc auc
    #roc_value = roc_auc_score(test_labels, predictions)
    # print(roc_value)
def svm_test_one_sample(test_data):
    clf = pickle.load(open('svm_model.sav', 'rb'))

    return clf.predict(test_data)
