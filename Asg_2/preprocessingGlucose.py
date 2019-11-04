import glob
import csv

import sys
# change float to int for series lunch files
path = sys.argv[1]


for fname in glob.glob(path):
	# print fname
	with open(fname) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		print fname
		# n = len(readCSV[0])
		rows=[]
		current = 0
		for row in readCSV:
			current=0
			count2=0
			count=0
			n = len(row)
		
			if(row[0]=='NaN' or row[0] == ''):
				while( current< n and (row[current] =='NaN' or row[current] =='') ):
					# print current
					current+=1

			if(current == n):
				continue

			if(current< n  and (row[current]!='' or row[current]!='NaN')):
				temp = row[current]
			else:
				break
	
			temp.strip()
			for i in range(0,n):
				if(row[i] == 'NaN' or row[i]==''):
					count2+=1		
					row[i]=temp
				else:
					temp = row[i]

			
			if(count2<25):
				rows.append(row)
			count+=1
			
			# print row[n-1]

	with open(fname+'_updated.csv',mode='w') as file:
		writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		# writer.writerow(header)
		for row in rows:
			writer.writerow(row)
	

# 2,11,26,74 deleted. form 3rd person