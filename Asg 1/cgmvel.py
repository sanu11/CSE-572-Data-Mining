import pandas as pd
import matplotlib.pyplot as plt

def velocity():
    fl = pd.read_csv("CGMSeriesLunchPat1.csv")
    dt = pd.read_csv("CGMDatenumLunchPat1.csv")
    c=0
    final = [[0 for i in range(3)] for j in range(fl.shape[0])]
    for row in range(fl.shape[0]):
        tm = dt.loc[row]
        cgm = fl.loc[row]
        v=[]
        for i in range(len(cgm)):
            if i == 0:
                v.append(0)
            else:
                v.append((cgm[i]-cgm[i-1])/(tm[i]-tm[i-1]))
        """plt.figure("CGM Velocity")
        plt.plot(dt.loc[row],v)
        plt.xlabel("Time Series")
        plt.title("Features")
        plt.show()"""
        for j in range(3):
            l= max(v)
            final[row][j] = l
            v.remove(l)
    return (final)
    
