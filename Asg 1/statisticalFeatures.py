from pandas import read_csv
from matplotlib import pyplot
import numpy as np
import pandas as pd


file_name = 'Datasets/CGMSeriesLunchPat1.csv'
def calculateWindowedMean(dataframe, window_size, window_slide):
    mean_matrix = []
    for i in range(0, len(dataframe)):
        row_mean = []

        st = 0
        end = st+window_size
        while end<=33:
            if end>29:
                window =  dataframe.iloc[i,st:30]
                #print("last window ",window)
                mean = np.mean(window)
                row_mean.append(mean)
                break
            #print("st end",st,end)
            window = dataframe.iloc[i,st:end]
            #print("window values ",window)
            mean = np.mean(window)
            row_mean.append(mean)
            st = st + window_slide
            end = st + window_size

        mean_matrix.append(row_mean)

    #print("rows",len(mean_matrix))
    #print("cols", len(mean_matrix[0]))
    #print(mean_matrix)
    return mean_matrix




def calculateWindowedstd(dataframe, window_size, window_slide):
    std_matrix = []
    for i in range(0, len(dataframe)):
        row_std = []

        st = 0
        end = st+window_size
        while end<=33:
            if end>29:
                window =  dataframe.iloc[i,st:30]
                #print("last window ",window)
                std = np.std(window)
                row_std.append(std)
                break
            #print("st end",st,end)
            window = dataframe.iloc[i,st:end]
            #print("window values ",window)
            std = np.std(window)
            row_std.append(std)
            st = st + window_slide
            end = st + window_size

        std_matrix.append(row_std)

    #print("rows",len(mean_matrix))
    #print("cols", len(mean_matrix[0]))
    #print(std_matrix)
    return std_matrix



def calculateWindowedMinMax(dataframe, window_size, window_slide):
    min_matrix = []
    max_matrix = []
    for i in range(0, len(dataframe)):
        row_min = []
        row_max = []

        st = 0
        end = st+window_size
        while end<=33:
            if end>29:
                window =  dataframe.iloc[i,st:30]
                #print("last window ",window)
                minn = min(window)
                maxx = max(window)
                row_min.append(minn)
                row_max.append(maxx)
                break
            #print("st end",st,end)
            window = dataframe.iloc[i,st:end]
            #print("window values ",window)
            minn = min(window)
            maxx = max(window)
            row_min.append(minn)
            row_max.append(maxx)
            st = st + window_slide
            end = st + window_size

        min_matrix.append(row_min)
        max_matrix.append(row_max)

    #print("rows",len(mean_matrix))
    #print("cols", len(mean_matrix[0]))
    #print(std_matrix)
    return min_matrix, max_matrix


def getStatisticalFeatures(file_name):
    dataframe = read_csv(file_name, header=0, index_col=False)
    print(dataframe)

    window_size = 9
    window_slide = 6

    mean_matrix = calculateWindowedMean(dataframe, window_size, window_slide)
    std_matrix = calculateWindowedstd(dataframe, window_size, window_slide)
    min_matrix, max_matrix = calculateWindowedMinMax(dataframe, window_size, window_size)



getStatisticalFeatures(file_name)



