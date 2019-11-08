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
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

def decisiontest(train_data, train_label):
    dt = DecisionTreeClassifier()
    dt.fit(train_data, train_label)
    #print("Tre:-----------------------------------------------------------------------------------")
    #print(dt)
    #print("Plotting tree: ---------------------------------------------------------------------")
    #tree.plot_tree(dt)
    # dot_data = StringIO()
    # export_graphviz(dt, out_file=dot_data, filled=True, rounded=True, special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # Image(graph.create_png())
    pickle.dump(dt,open('dt_model.sav','wb'))

def decisiontrain(test_data,test_label):
    dt = pickle.load(open('dt_model.sav', 'rb'))
    predictions = dt.predict(test_data)
    print(accuracy_score(test_label,predictions))

train_X,train_Y,test_X, test_Y = pcafeature.get_feature_matrix_Final('Asg_2/Data/concatenatedData.csv')
decisiontest(train_X, train_Y)
decisiontrain(test_X,test_Y)


