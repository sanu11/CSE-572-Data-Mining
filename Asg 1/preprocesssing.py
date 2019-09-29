import glob
import csv


# change float to int for series lunch files
path = "./DataSets/CGMDatenumLunch*.csv"
for fname in glob.glob(path):
	print fname
	with open(fname) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		print readCSV
		# n = len(readCSV[0])
		rows=[]
		count=0
		for row in readCSV:
			if count!=0:
				summ=0
				# print row[n-1]
				missing=[]
			
				for i in range(1,n):
					if(row[i].strip()!='' and row[i-1].strip()!=''):
						temp = float(row[i])-float(row[i-1])
						summ+=temp
					elif(row[i].strip()==''):
						missing.append(i)
				avg=summ/n
				if(len(missing)!=0):
					# print missing
					for j in missing:
						row[j] = float(row[j-1])+avg
						# print row[j]

				
			else:
				# header = row
				n =len(row)
				count+=1
			rows.append(row)
			# print row[n-1]

	with open(fname+'_updated.csv',mode='w') as file:
		writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		# writer.writerow(header)
		for row in rows:
			writer.writerow(row)
	