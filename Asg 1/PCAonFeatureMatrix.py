from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.fftpack import fft, ifft

#FinalFeatureFrame = pd.concat([meanFrame,powerFrame,rmsFrame], axis = 1)
#X = StandardScaler().fit_transform(FinalFeatureFrame)
#READ final feature matrix from file
FinalFeatureFrame = read_csv('FeatureMatrix.csv',header=0)

#perform PCA
pca = PCA()
x_new = pca.fit_transform(FinalFeatureFrame)
#create dataframe from pca components
PCAframe = pd.DataFrame(x_new)
#save to file
PCAframe.to_csv('PCAData.csv', index = None)
print(pca.explained_variance_ratio_)

for i in pca.explained_variance_ratio_:
  print("PCA Components and their variance",i)
