import pywt
from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.fftpack import fft, ifft
import statisticalFeatures as statf
import cgmvel as cgm

final_file = 'Datasets/CGMSeries_CombinedFile.csv'
#Plot all data points
# dataframe2 = read_csv(final_file, header=0, index_col=False)
# dataframe2 = dataframe2.iloc[:,:31]
dataframe_time = read_csv(final_file, header=0, index_col=False);
dataframe_time = dataframe_time.iloc[:,:31];
#Reversing the CGM Series
dataframe_time = dataframe_time.iloc[:,::-1];
num_rows = len(dataframe_time);
num_cols = len(dataframe_time.iloc[0]);
print("No. of rows in our total time series dataset = ",num_rows)
print("No. of cols in our total time series dataset = ",num_cols)
print("DataFrame is reversed")

coeffs = pywt.dwt2(dataframe_time, 'db1',mode='symmetric')
cA, (cH, cV, cD) = coeffs
print len(cA[0]),len(cH[0]),len(cV[0]),len(cD[0])
print len(cA),len(cH),len(cV),len(cD)


for i in range(0,len(dataframe_time)):

	d=dataframe_time.iloc[i,0:31]
	(cA,cH) = pywt.dwt(d, 'db1',mode='symmetric')
	# print coeffs
	N = len(dataframe_time)
	# sample spacing
	T = 1.0 / 36.0
	x = np.linspace(0.0, N*T, N)

	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
	sorted(yf,reverse=True)

	print("now fft")

	freq1_ps = np.abs(yf)**2
	power.append(np.sum(freq1_ps))
	peaks.append(np.abs(yf[0:4]))

	listOfMeans.append(np.mean(list))
	std.append(np.std(list))
      #print(listOfMeans)
      #pyplot.hist(listOfMeans)

# powerFrame = pd.DataFrame.from_dict(power)
# peaksFrame = pd.DataFrame.from_dict(peaks)
# pyplot.xlabel('Data')
# pyplot.ylabel('FFT Peak1')
# pyplot.scatter(range(0,N),(np.array(peaks))[:,0])
# pyplot.savefig('Plots/Plots_fft/PCA_fft_peak1')
# pyplot.show()
