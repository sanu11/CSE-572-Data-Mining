from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd
import numpy as np
import pickle
import PCAonFeatureMatrix as pcafeature
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.model_selection import cross_val_score

def decisiontest(train, label):
    dt = DecisionTreeClassifier(random_state = 0)
    kf = KFold(n_splits=6, shuffle = True)
    ascore=[]
    pr=[]
    rc=[]
    f1=[]
    for train_index, test_index in kf.split(train, label):
        train_data, train_l = train.iloc[train_index], label.iloc[train_index]
        test_data, test_l = train.iloc[test_index], label.iloc[test_index]
        dt.fit(train_data, train_l)
        predictions = dt.predict(test_data)
        a = accuracy_score(predictions,test_l)
        ascore.append(a)
        pr.append(precision_score(predictions, test_l))
        rc.append(recall_score(predictions, test_l))
        f1.append(f1_score(predictions, test_l))
    #print("k=",j)
    print("Accuracy:", ascore)
    print("Mean Accuracy: ", sum(ascore)/len(ascore))
    print("Precision: ",sum(pr)/len(pr))
    print("Recall:", sum(rc)/len(rc))
    print("F1 Score: ",sum(f1)/len(f1))
    #print("---------------------------------------------------------------------------------------------")
    pickle.dump(dt,open('dt_model.sav','wb'))

    # print("Printing score:---------------------------------------------------------------------------------------")
    # for i in range(2,11):
    #     print("i = ",i)
    #     s= cross_val_score(dt, train, label, cv=i, scoring='accuracy')
    #     print("Score=",s.mean())
    #     print("-----------------------------------------------------------------------------------------------------")

def decisiontrain(test_data,test_label):
    dt = pickle.load(open('dt_model.sav', 'rb'))
    predictions = dt.predict(test_data)
    print("Decision Tree Accuracy: ", accuracy_score(test_label,predictions))
    print("Decision Tree precision = ",precision_score(test_label,predictions))
    print("Decision Tree recall = ",recall_score(test_label,predictions))
    print("Decision Tree F1 = ",f1_score(test_label,predictions))
    print("Decision Tree Confusion Matrix :")
    print(confusion_matrix(predictions, test_label))


def dt_test_one(test_data):
    dt = pickle.load(open('dt_model.sav', 'rb'))
    predictions = dt.predict(test_data)
    print("Predicted class:", predictions)
    return predictions

#data = pcafeature.get_feature_matrix_Final('Asg_2/Data/concatenatedData.csv')
#decisiontest(data)



