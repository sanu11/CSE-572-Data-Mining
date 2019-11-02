import glob
import csv


# change float to int for series lunch files
path = "./DataSets/CGMDatenumLunchPat*.csv"
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
			count2=0
			if count!=0:
				summ=0
				# print row[n-1]
				missing=[]
			
				for i in range(1,n):
					if(row[i].strip()!='' and row[i-1].strip()!='' and row[i]!='NaN' and row[i-1]!='NaN'):
						# print row[i],row[i-1]
						temp = float(row[i])-float(row[i-1])
						summ+=temp
						count2+=1
					else:
						missing.append(i)
				avg=summ/count2
				missing.sort()
				# print row
				# print len(missing)
				# print count,summ,count2,avg
				if(len(missing)!=0 and len(missing)!=n and len(missing)!=41):
					# print missing
					for j in missing:
						# print j,j-1,row[j-1],avg

						row[j] = float(row[j-1])+avg
						# print row[j]

				
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
	