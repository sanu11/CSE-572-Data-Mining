import pandas as pd
import matplotlib.pyplot as plt

def velocity(cgmSeries, dataenum):
    # fl = pd.read_csv("DataSets/CGMSeriesLunchPat1.csv")
    # dt = pd.read_csv("Datasets/CGMDatenumLunchPat1.csv")
    fl = cgmSeries
    dt = dataenum
    c=0
    df = pd.DataFrame(index=range(0,fl.shape[0]),columns=['velocity1','velocity2','velocity3'])
    final = [[0 for i in range(3)] for j in range(fl.shape[0])]
    for row in range(fl.shape[0]):
        #tm = dt.loc[row]
        cgm = fl.loc[row]
        v=[]
        for i in range(len(cgm)):
            if i == 0:
                v.append(0)
            else:
                #v.append(abs(cgm[i]-cgm[i-1])/abs(tm[i]-tm[i-1]))
                v.append(abs(cgm[i]-cgm[i-1]))
        """plt.figure("CGM Velocity")
        plt.plot(dt.loc[row],v)
        plt.xlabel("Time Series")
        plt.title("Features")
        plt.show()"""
        for j in range(3):
            ma= max(v)
            final[row][j] = ma
            v.remove(ma)
            print(final[row][j])

    x = list(zip(*final))[0]
    l = list(range(0,fl.shape[0]))
    print(v)
    for i in range(0,3):
        # plt.scatter(l,list(zip(*final))[i])
        # plt.xlabel("Data")
        # plt.ylabel("CGM Velocity "+str(i+1))
        # plt.title("CGM velocity")
        # plt.savefig('Plots/Plots_Cgm/PCA_cgm'+str(i+1))
        # plt.show()
        df = pd.DataFrame(final)
    return (df)
