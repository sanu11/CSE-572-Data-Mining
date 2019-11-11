# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import pickle
import numpy as np
RSEED = 50
def random_forest_train(train_features,train_labels):
    print("Starting RandomForest Training Model")
    # Instantiate model with 1000 decision trees
    # Using K-Fold to get Samples
    rf = RandomForestClassifier(n_estimators=100)
    kfold_accuracies = []
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(train_features):
        X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
        Y_train, Y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
        # Train the model on training data
        rf.fit(X_train,Y_train)
        kfold_accuracies.append(accuracy_score(rf.predict(X_test), Y_test))
    print("\n\n***********Random Forest Output******************")
    print("Random Forest Accuracies K Fold: ",kfold_accuracies)
    print("Random Forest Mean Accuracy(K fold): ", sum(kfold_accuracies)/len(kfold_accuracies))
    pickle.dump(rf,open('rf_model.sav','wb'))
    # Use the forest's predict method on the test data

def random_forest_test(test_features,test_labels):
    rf = pickle.load(open('rf_model.sav', 'rb'))

    predictions = rf.predict(test_features)
    print("**************Test Accuracy****************")

    print("Random Forest accuracy = ",accuracy_score(test_labels,predictions))
    print("Random Forest precision = ",precision_score(test_labels,predictions))
    print("Random Forest recall = ",recall_score(test_labels,predictions))
    print("Random Forest F1 = ",f1_score(test_labels,predictions))
    print("\n")
    # Calculate roc auc
    #roc_value = roc_auc_score(test_labels, predictions)
    # print(roc_value)
def rf_test_one_sample(test_data):
    rf = pickle.load(open('rf_model.sav', 'rb'))

    return rf.predict(test_data)
