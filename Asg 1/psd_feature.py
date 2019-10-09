import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import numpy as np

def psd(cgmSeries):
    #cgmSeries, dataenum
    #fl = pd.read_csv("C:/Users/user/Desktop/CSE-572-Data-Mining-master/Asg 1/DataSets/CGMSeries_CombinedFile.csv")
    fl = cgmSeries
    df = pd.DataFrame(index=range(0,fl.shape[0]),columns=['PSD1','PSD2','PSD3'])
    p = [[0 for i in range(3)] for j in range(fl.shape[0])]
    for row in range(fl.shape[0]):
        r = fl.loc[row]
        v,u = (scipy.signal.periodogram(r))
        u1 = list(u)
        print(u1)
        for j in range(3):
            m= max(u1)
            p[row][j] = m
            u1.remove(m)
    df = pd.DataFrame(p)
    l = list(range(1,fl.shape[0]+1))
    for i in range(0,3):
        plt.scatter(l,list(zip(*p))[i])
        plt.xlabel("Data")
        plt.ylabel("Power Spectral Density "+str(i+1))
        plt.title("Power Spectral Density")
        plt.savefig('Plots/Plots_psd/PCA_psd'+str(i+1))
        plt.show()
    return df
