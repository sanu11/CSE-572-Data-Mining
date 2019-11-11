
from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.fftpack import fft, ifft
import FeatureExtractionFFT as feature_matrix
from sklearn.model_selection import train_test_split
from numpy import genfromtxt

savetop5EigenTranspose=0

def get_feature_matrix_Final(filename):
    #READ final feature matrix from file`
    #FinalFeatureFrame = read_csv('FeatureMatrix.csv',header=0)
    data_File = filename
    labels = read_csv(data_File,header=0)
    len_of_file= len(labels)

    final_labels = labels.iloc[0:len_of_file,30]
    # print(final_labels)

    FinalFeatureFrame = feature_matrix.feature_matrix_for_pca(data_File)
    FinalFeatureFrame.drop(FinalFeatureFrame.columns[1], axis=1,inplace=True)
    N= len(FinalFeatureFrame)

    #perform PCA
    pca = PCA()
    x_new = pca.fit_transform(FinalFeatureFrame) 
    # print(len(x_new))
    #create dataframe from pca components
    PCAframe = pd.DataFrame(x_new)
    #save to file
    #PCAframe.to_csv('PCAData.csv', index = None)
    #get the top 5 eigen from pca
    top5Eigen = np.array(pca.components_)[0:5][0:len(pca.components_)]
    #transpose to get the eigen vectors in columns
    top5EigenTranspose = np.transpose(top5Eigen)
    # outfile = TemporaryFile()
    # np.save('outfile', top5EigenTranspose)

    #print(len(top5EigenTranspose))
    #savetop5EigenTranspose = top5EigenTranspose

    #calculate the final feature matrix by multiplying
    finalFeatureMatrix = FinalFeatureFrame.dot(top5EigenTranspose)
    pd.DataFrame(top5EigenTranspose).to_csv('Top5EigenVectors.csv')
    #print(finalFeatureMatrix)
    #pyplot.scatter(range(0,N),np.array(FinalFeatureFrame)[0:N,0])
    pyplot.xlabel('Data')
    pyplot.ylabel('PCA Feature 1')
    pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][0])
    #pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature1.png')

    pyplot.xlabel('Data')
    pyplot.ylabel('PCA Feature 2')
    pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][1])
    #pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature2.png')

    pyplot.xlabel('Data')
    pyplot.ylabel('PCA Feature 3')
    pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][2])
    #pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature3.png')

    pyplot.xlabel('Data')
    pyplot.ylabel('PCA Feature 4')
    pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][3])
    #pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature4.png')
    #pyplot.show()
    pyplot.close()

    pyplot.xlabel('Data')
    pyplot.ylabel('PCA Feature 5')
    pyplot.scatter(range(0,N),finalFeatureMatrix[0:N][4])
    #pyplot.savefig('Plots/Plots_after_PCA/PCA_Feature5.png')
    pyplot.close()
    #pyplot.show()

    #pcaFeatureImp = pd.DataFrame(pca.components_[0:5],columns=range(0,len(pca.components_)),index = ['PC-1','PC-2','3','4','5'])
    #pcaFeatureImp.to_csv('pcaFeatureImp.csv')
    #print(pca.explained_variance_ratio_)
    #print(pcaFeatureImp.shape)

    final_matrix_for_models = pd.DataFrame(finalFeatureMatrix)
    # print(final_matrix_for_models.shape)
    final_matrix_for_models = pd.concat([final_matrix_for_models,final_labels],axis=1)

    #final_matrix_for_models.concat(final_labels)
    #print(final_matrix_for_models.shape)
    final_matrix_for_models.to_csv('final_matrix_for_models.csv')
    x_train ,x_test = train_test_split(final_matrix_for_models,test_size=0.2)       #test_size=0.5(whole_data)

    return final_matrix_for_models.iloc[:,0:5],final_matrix_for_models.iloc[:,5],x_test.iloc[:,0:5],x_test.iloc[:,5]

def get_reduced_test_data(test_data):

    top5EigenTranspose=genfromtxt('Top5EigenVectors.csv', delimiter=',')
    top5EigenTranspose=top5EigenTranspose[1:]
    top5EigenTranspose =top5EigenTranspose[:,1:]

    test_data = test_data.dot(top5EigenTranspose)
    return test_data
