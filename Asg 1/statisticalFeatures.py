from pandas import read_csv
from matplotlib import pyplot
import numpy as np
import pandas as pd


file_name = 'Datasets/CGMSeriesLunchPat1.csv'
def calculateWindowedMean(dataframe, window_size, window_slide ):
    pass


def calculateWindowedstd():
    pass



def calculateWindowedMinMax():
    pass



def getStatisticalFeatures(file_name):
    dataframe = read_csv(file_name, header=0, index_col=False)
    print(dataframe)

    window_size = 9
    window_slide = 6
    
    mean_matrix = calculateWindowedMean(dataframe, window_size, window_slide)
    std_matrix = calculateWindowedstd(dataframe, window_size, window_slide)
    min_matrix, max_matrix = calculateWindowedMinMax(dataframe, window_size, window_size)



getStatisticalFeatures(file_name)



