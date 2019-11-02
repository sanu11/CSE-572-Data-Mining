import glob
import csv


# change float to int for series lunch files
path = "./DataSets/CGMSeriesLunch*.csv"
for fname in glob.glob(path):
	# print fname
	with open(fname) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		print readCSV
		# n = len(readCSV[0])
		rows=[]
		count=0
		count2=0
		for row in readCSV:
			
			if count!=0:
				temp = row[0]
				temp.strip()
				for i in range(0,n):
					if(row[i] == 'NaN' or row[i]==''):
						if(temp!='NaN' or temp!=''):
							row[i]=temp
							# temp=row[i]
					temp = row[i]
				
			else:
				# header = row
				n =len(row)
				# count=1
			rows.append(row)
			count+=1
			# print row[n-1]

	with open(fname+'_updated.csv',mode='w') as file:
		writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		# writer.writerow(header)
		for row in rows:
			writer.writerow(row)
	

# 2,11,26,74 deleted. form 3rd person