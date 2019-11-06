import numpy as np
from sklearn.model_selection import KFold

#from numpy import genfromtxt
mydata = np.genfromtxt('Asg_2\Data\mealData1_updated.csv', delimiter=',')
kf = KFold(n_splits=2, shuffle= True)
for train_index, test_index in kf.split(mydata):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = mydata[train_index], mydata[test_index]
    #print("Training dta:", X_train)
    #print("Testing data:", X_test)