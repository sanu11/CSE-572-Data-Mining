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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def decisiontest(data):
    train=data.iloc[:,0:5]
    label = data.iloc[:,5]
    dt = DecisionTreeClassifier()
    kf = KFold(n_splits=5)
    i=1
    ascore=[]
    for train_index, test_index in kf.split(train, label):
        train_data, train_l = train.iloc[train_index], label.iloc[train_index]
        test_data, test_l = train.iloc[test_index], label.iloc[test_index]
        dt.fit(train_data, train_l)
        predictions = dt.predict(test_data)
        a = accuracy_score(predictions,test_l)
        ascore.append(a)
        print("Accuracy ",i," :", a)
        i+=1
    print("Mean Accuracy: ", sum(ascore)/len(ascore))
    pickle.dump(dt,open('dt_model.sav','wb'))

    # print("Printing score:---------------------------------------------------------------------------------------")
    # for i in range(2,11):
    #     print("i = ",i)
    #     s= cross_val_score(dt, train, label, cv=i, scoring='accuracy')
    #     print("Score=",s.mean())
    #     print("-----------------------------------------------------------------------------------------------------")
    

    #print("Plotting tree: ---------------------------------------------------------------------")
    #tree.plot_tree(dt)
    # dot_data = StringIO()
    # export_graphviz(dt, out_file=dot_data, filled=True, rounded=True, special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # Image(graph.create_png())

def decisiontrain(test_data,test_label):
    dt = pickle.load(open('dt_model.sav', 'rb'))
    predictions = dt.predict(test_data)
    print(accuracy_score(test_label,predictions))

def dt_test_one(test_data):
    df = pickle.load(open('dt_model.sav', 'rb'))
    predictions = dt.predict(test_data)
    print("Predicted class:", predictions)
    return predictions

#train_X,train_Y,test_X, test_Y = 
#data = pd.read_csv("final_matrix_for_models.csv") 
data = pcafeature.get_feature_matrix_Final('Asg_2/Data/concatenatedData.csv')
decisiontest(data)
#decisiontrain(test_X,test_Y)


