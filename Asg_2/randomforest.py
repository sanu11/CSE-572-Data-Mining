# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import pickle
import numpy as np
RSEED = 50
def random_forest_train(train_features,train_labels):
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=100)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    pickle.dump(rf,open('rf_model.sav','wb'))
    # Use the forest's predict method on the test data

def random_forest_test(test_features,test_labels):
    rf = pickle.load(open('rf_model.sav', 'rb'))

    predictions = rf.predict(test_features)

    print("Random Forest accuracy = ",accuracy_score(test_labels,predictions))
    print("Random Forest precision = ",precision_score(test_labels,predictions))
    print("Random Forest recall = ",recall_score(test_labels,predictions))
    print("Random Forest F1 = ",f1_score(test_labels,predictions))

    # Calculate roc auc
    #roc_value = roc_auc_score(test_labels, predictions)
    # print(roc_value)
def rf_test_one_sample(test_data):
    rf = pickle.load(open('rf_model.sav', 'rb'))

    return rf.predict(test_data)
