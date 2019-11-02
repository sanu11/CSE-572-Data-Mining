import numpy
import statistics
from numpy import genfromtxt


data = genfromtxt('CGMSeriesLunchPat1.csv', delimiter = ',', skip_header = 1)
st=[[0 for x in range(13)] for y in range(data.shape[0])]
window = 8
skip = 4
i=0
for row in data:
    j=0
    l=0
    while ((l+window)<row.size):
        stdev = numpy.std(row[l:l+window], axis=0)
        st[i][j] = stdev
        j=j+1
        m = numpy.mean(row[l:l+window]**2)
        rms = numpy.sqrt(m)
        st[i][j] = rms
        j=j+1
        l=l+skip
    mx = max(row[l:l+window])
    mn = min(row[l:l+window])
    diff = mx-mn
    st[i][j]=diff
    i=i+1
print (st)
