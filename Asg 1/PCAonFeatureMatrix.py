from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.fftpack import fft, ifft

#READ final feature matrix from file`
FinalFeatureFrame = read_csv('FeatureMatrix.csv',header=0)
FinalFeatureFrame.drop(FinalFeatureFrame.columns[1], axis=1,inplace=True)
N= len(FinalFeatureFrame)
#perform PCA
pca = PCA()
x_new = pca.fit_transform(FinalFeatureFrame)
# print(len(x_new))
#create dataframe from pca components
PCAframe = pd.DataFrame(x_new)
#save to file
PCAframe.to_csv('PCAData.csv', index = None)
#get the top 5 eigen from pca
top5Eigen = np.array(pca.components_)[0:5][0:len(pca.components_)]

#transpose to get the eigen vectors in columns
top5EigenTranspose = np.transpose(top5Eigen)
print(len(top5EigenTranspose))

#calculate the final feature matrix by multiplying
finalFeatureMatrix = FinalFeatureFrame.dot(top5EigenTranspose)
print(finalFeatureMatrix)
#pyplot.scatter(range(0,N),np.array(FinalFeatureFrame)[0:N,0])
pyplot.xlabel('Data')
pyplot.ylabel('PCA Feature 1')
pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][0])
pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature1.png')
pyplot.show()

pyplot.xlabel('Data')
pyplot.ylabel('PCA Feature 2')
pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][1])
pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature2.png')
pyplot.show()

pyplot.xlabel('Data')
pyplot.ylabel('PCA Feature 3')
pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][2])
pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature3.png')
pyplot.show()

pyplot.xlabel('Data')
pyplot.ylabel('PCA Feature 4')
pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][3])
pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature4.png')
pyplot.show()

pyplot.xlabel('Data')
pyplot.ylabel('PCA Feature 5')
pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][4])
pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature5.png')

pyplot.show()
