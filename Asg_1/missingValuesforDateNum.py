import glob
import csv


# change float to int for series lunch files
path = "./DataSets/CGMDatenumLunchPat5.csv"
for fname in glob.glob(path):
	print fname
	with open(fname) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		# print readCSV
		# n = len(readCSV[0])
		rows=[]
		# count=0
		# count2=0
		for row in readCSV:
			# count2=0
			n = len(row)
			for i in range(1,n):
				if(row[i].strip()=='' or row[i]=='NaN'):
					# print row[i],row[i-1]
					if(str(row[i-1]).strip()!='' and str(row[i-1])!='NaN'):
						row[i] = float(row[i-1])-0.003472222
					else:
						print "didnot update ",i
				
			rows.append(row)
		
			# print row[n-1]

	with open(fname+'_updated.csv',mode='w') as file:
		writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		# writer.writerow(header)
		for row in rows:
			writer.writerow(row)
	