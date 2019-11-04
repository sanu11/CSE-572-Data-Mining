import glob
import csv
path = "Data/*_updated.csv"
rows=[]
count=0
for fname in glob.glob(path):
	print fname
	with open(fname) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			count+=1
	print count
