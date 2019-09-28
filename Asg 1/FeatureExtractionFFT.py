from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.fftpack import fft, ifft
import statisticalFeatures as statf
# load dataset
#dataframe1 = read_csv('CGMDatenumLunchPat1.csv',header=0, names=['cgmDatenum_1'])
#print(series1.head(5))
dataframe2 = read_csv('Datasets/CGMSeriesLunchPat1.csv', header=0, index_col=False)
#dataframe2.reindex(dataframe1)
# display first few rows
#print(dataframe2.head(5))
# line plot of dataset
dataframe2.plot()
pyplot.show()
listOfMeans=[]
std = []
power =[]
peaks = []
rms=[]
for i in range(0,len(dataframe2)):
  #n=9

  #print(len(dataframe2))
  #k=6
  #p=0
  #print(dataframe2.loc[i,:])
  list=dataframe2.iloc[i,0:30]
  #print(list)
  #print(np.isnan(list))
 # list = list
  y= fft(list)
  N = 33
  # sample spacing
  T = 1.0 / 36.0
  x = np.linspace(0.0, N*T, N)
  #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
  yf = fft(y)
  xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
  sorted(list,reverse=True)
  #pyplot.plot(list)
  #pyplot.show()
  #pyplot.clear()
  #pyplot.clf()
  #fig, ax = pyplot.subplots()
  print("now fft")
  #ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
  #ax.plot(xf, fftfreq(yf))
  #pyplot.show()
  #print(y)

  freq1_ps = np.abs(yf)**2
  power.append(np.sum(freq1_ps))
  peaks.append(np.abs(yf[0:4]))
  print("power =")
  print(power)
 # pyplot.clf()
  #pow1 = yf.*conj(yf)/(30);
  #sum = sum + pow1
  #slice = []
  #print(len(list)/k)
  #for k in range(0,len(list)/k):
   # for j in range(p,n):
    #  if j<len(list):
     #   slice.append(list[j])
    #listOfMeans.append(np.mean(slice))
    #p=p+6
    #j=j+n
  listOfMeans.append(np.mean(list))
  std.append(np.std(list))
      #print(listOfMeans)
      #pyplot.hist(listOfMeans)

#meanFrame = pd.DataFrame.from_dict(listOfMeans)
#stdFrame = pd.DataFrame.from_dict(std)
powerFrame = pd.DataFrame.from_dict(power)
peaksFrame = pd.DataFrame.from_dict(peaks)
mean_matrix,std_matrix,min_matrix,max_matrix = statf.getStatisticalFeatures('Datasets/CGMSeriesLunchPat1.csv')
meanFrame = pd.DataFrame.from_dict(mean_matrix)
stdFrame = pd.DataFrame.from_dict(std_matrix)
minFrame = pd.DataFrame.from_dict(min_matrix)
maxFrame = pd.DataFrame.from_dict(max_matrix)


#pyplot.plot(range(0,33), power)
print(len(listOfMeans))
FinalFeatureFrame = pd.concat([peaksFrame,meanFrame,stdFrame,minFrame,maxFrame], axis = 1)
print(FinalFeatureFrame)

FinalFeatureFrame.to_csv('FeatureMatrix.csv')
#np.save('FeatureMatrix.csv',FinalFeatureFrame)
#X = StandardScaler().fit_transform(FinalFeatureFrame)

