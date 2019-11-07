# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import pickle
import numpy as np
RSEED = 50
def random_forest_train(train_features,train_labels):
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=100,
                               random_state=RSEED,
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    pickle.dump(rf,open('rf_model.sav','wb'))
    # Use the forest's predict method on the test data

def random_forest_test(test_features,test_labels):
    rf = pickle.load(open('rf_model.sav', 'rb'))

    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    # errors = abs(predictions - test_labels)
    # # Print out the mean absolute error (mae)
    # print(errors,test_labels)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #
    # # Calculate mean absolute percentage error (MAPE)
    # mape = 100 * (errors / test_labels)
    #
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')
    # result = rf.score(test_features, test_labels)
    # print(result)
    #print(predictions)
    print(accuracy_score(test_labels,predictions))

    # Calculate roc auc
    #roc_value = roc_auc_score(test_labels, predictions)
    # print(roc_value)
def rf_test_one_sample(test_data):
    rf = pickle.load(open('rf_model.sav', 'rb'))

    return rf.predict(test_data)
