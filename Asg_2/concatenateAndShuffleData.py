import glob
import csv
import random
path = "Data/*_updated.csv"
rows=[]
count=0
for fname in glob.glob(path):
	print fname
	with open(fname) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			rows.append(row)

print len(rows)

random.shuffle(rows)

with open('concatenatedData.csv',mode='w') as file:
		writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		# writer.writerow(header)
		for row in rows:
			writer.writerow(row)
	
