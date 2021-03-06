from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.fftpack import fft, ifft
import statisticalFeaturesForTest as statf
import cgmvel as cgm
import psd_feature as psd
# load dataset
#dataframe1 = read_csv('CGMDatenumLunchPat1.csv',header=0, names=['cgmDatenum_1'])
#print(series1.head(5))

#print("Get all Files\n")
#all_filenames = ['Datasets/CGMSeriesLunchPat1.csv_updated.csv','Datasets/CGMSeriesLunchPat2.csv_updated.csv','Datasets/CGMSeriesLunchPat3.csv_updated.csv','Datasets/CGMSeriesLunchPat4.csv_updated.csv','Datasets/CGMSeriesLunchPat5.csv_updated.csv']
#final_file = 'Datasets/CGMSeries_CombinedFile.csv'
#combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames],sort=False)
#print("Combined all files\n")
#combined_csv.to_csv(final_file, index=False, encoding='utf-8-sig')
#print("Wrote all files to final file")
def feature_matrix_for_pca(concatenatedFile):
  data_File = concatenatedFile
  # combined_csv = 'Data/concatenatedData.csv'
  final_file =concatenatedFile
  #Plot all data points
  dataframe2 = read_csv(data_File, header=None, index_col=False)
  dataframe2 = dataframe2.iloc[:,:30]
  # print concatenatedFile
  #Reversing the CGM Series
  dataframe2 = dataframe2.iloc[:,::-1]
  num_rows = len(dataframe2)
  num_cols = len(dataframe2.iloc[0])
  # print("No. of rows in our total dataset = ",num_rows)
  # print("No. of cols in our total dataset = ",num_cols)
  # print("DataFrame is reversed")


  dataframe_time = read_csv(final_file, header=None, index_col=False)
  dataframe_time = dataframe_time.iloc[:,:31]
  #Reversing the CGM Series
  dataframe_time = dataframe_time.iloc[:,::-1]
  num_rows = len(dataframe_time)
  num_cols = len(dataframe_time.iloc[0])
  # print("No. of rows in our total time series dataset = ",num_rows)
  # print("No. of cols in our total time series dataset = ",num_cols)
  # print("DataFrame is reversed")

  dataframe2.plot()
  #pyplot.show()
  listOfMeans=[]
  std = []
  power =[]
  peaks = []
  rms=[]
  for i in range(0,len(dataframe2)):

    list=dataframe2.iloc[i,0:31]

    yf= fft(list)
    N = len(dataframe2)
    # sample spacing
    T = 1.0 / 36.0
    x = np.linspace(0.0, N*T, N)

    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    sorted(yf,reverse=True)
    #print("now fft")

    freq1_ps = np.abs(yf)**2
    power.append(np.sum(freq1_ps))
    peaks.append(np.abs(yf[0:4]))

    listOfMeans.append(np.mean(list))
    std.append(np.std(list))
        #print(listOfMeans)
        #pyplot.hist(listOfMeans)

  powerFrame = pd.DataFrame.from_dict(power)
  peaksFrame = pd.DataFrame.from_dict(peaks)



  #get statistical features
  mean_matrix,std_matrix,min_matrix,max_matrix,diff_minmax = statf.getStatisticalFeatures(final_file)

  meanFrame = pd.DataFrame.from_dict(mean_matrix)
  stdFrame = pd.DataFrame.from_dict(std_matrix)
  minFrame = pd.DataFrame.from_dict(min_matrix)
  maxFrame = pd.DataFrame.from_dict(max_matrix)
  diffminmaxFrame = pd.DataFrame.from_dict(diff_minmax)

  #get cgm veocity peaks
  cgmVelocityFrame = cgm.velocity(dataframe2,dataframe_time)
  psdFrame = psd.psd(dataframe2)


  FinalFeatureFrame = pd.concat([peaksFrame,meanFrame,stdFrame,minFrame,maxFrame,diffminmaxFrame,cgmVelocityFrame,psdFrame], axis = 1)
  # print(FinalFeatureFrame.shape)

  #FinalFeatureFrame.to_csv('FeatureMatrix.csv')
  # print FinalFeatureFrame
  return FinalFeatureFrame
  #np.save('FeatureMatrix.csv',FinalFeatureFrame)
  #X = StandardScaler().fit_transform(FinalFeatureFrame)

