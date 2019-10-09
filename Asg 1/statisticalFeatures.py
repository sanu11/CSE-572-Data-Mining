from pandas import read_csv
from matplotlib import pyplot
import numpy as np
import pandas as pd
from matplotlib import pyplot


file_name = 'Datasets/CGMSeries_CombinedFile.csv'
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
    diff_minmax = []
    for i in range(0, len(dataframe)):
        row_min = []
        row_max = []
        row_diff = []

        st = 0
        end = st+window_size
        while end<=33:
            if end>30:
                window =  dataframe.iloc[i,st:31]
                #print("last window ",window)
                minn = min(window)
                maxx = max(window)
                row_min.append(minn)
                row_max.append(maxx)
                row_diff.append(maxx-minn)
                break
            #print("st end",st,end)
            window = dataframe.iloc[i,st:end]
            #print("window values ",window)
            minn = min(window)
            maxx = max(window)
            row_min.append(minn)
            row_max.append(maxx)
            row_diff.append(maxx - minn)
            st = st + window_slide
            end = st + window_size

        min_matrix.append(row_min)
        max_matrix.append(row_max)
        diff_minmax.append(row_diff)

    #print("rows",len(mean_matrix))
    #print("cols", len(mean_matrix[0]))
    #print(std_matrix)
    diff_min = []
    for i in range(0,len(min_matrix)):
        row_min = []
        for j in range(1,len(min_matrix[0])):
            row_min.append(min_matrix[i][j]-min_matrix[i][j-1])
        diff_min.append(row_min)

    diff_max = []
    for i in range(0,len(max_matrix)):
        row_max = []
        for j in range(1,len(max_matrix[0])):
            row_max.append(max_matrix[i][j]-max_matrix[i][j-1])
        diff_max.append(row_max)

    return diff_min, diff_max,diff_minmax


def plotfeatures(feature_matrix, subtitle, fname):

    n_rows = len(feature_matrix)
    n_cols = len(feature_matrix[0])
    print("Rows and cols in feature matrix ",n_rows,n_cols)

    colors = ['r','b','g','y','c']
    for i in range(0,n_cols):
        feature_values = []
        for j in range(0,n_rows):
            feature_values.append(feature_matrix[j][i])

        print(len(feature_values))
        #pyplot.subplot(123+i)
        pyplot.scatter(range(0,n_rows),feature_values, c ='b')

        pyplot.ylabel('Feature Value')
        pyplot.xlabel('Different Data sample')
        #pyplot.legend('12345')
        pyplot.suptitle("Scatterplot for Statistical Features")
        pyplot.title(subtitle+" window "+str(i))
        pyplot.show()
        pyplot.savefig(fname+"_"+str(i))





def getStatisticalFeatures(file_name):
    dataframe = read_csv(file_name, header=0, index_col=False)

    dataframe = dataframe.iloc[:,:31]
    print(dataframe)
    window_size = 9
    window_slide = 6

    mean_matrix = calculateWindowedMean(dataframe, window_size, window_slide)
    print("rows",len(mean_matrix))
    print("cols", len(mean_matrix[0]))
    print("mean matrix ",mean_matrix)
    std_matrix = calculateWindowedstd(dataframe, window_size, window_slide)
    print("std_matrix ", std_matrix)
    min_matrix, max_matrix, diff_minmax = calculateWindowedMinMax(dataframe, window_size, window_slide)
    print("min_matrix ", min_matrix)
    print("max_matrix ", max_matrix)

    # plotfeatures(mean_matrix, "Mean of of sliding windows (window size=9)","stat_fig1")
    # plotfeatures(std_matrix, "Std Deviation of sliding windows (window size=9)","stat_fig2")
    # plotfeatures(min_matrix, "Difference in Min Values of sliding windows","stat_fig3")
    # plotfeatures(max_matrix, "Difference in Max Values of sliding windows","stat_fig4")
    # plotfeatures(diff_minmax, "Difference in Max Values of sliding windows","stat_fig5")

    return mean_matrix,std_matrix,min_matrix,max_matrix,diff_minmax



getStatisticalFeatures(file_name)



