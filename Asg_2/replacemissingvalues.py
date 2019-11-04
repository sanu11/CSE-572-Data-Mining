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
		current=0
		count2=0
		count=0
		rowcount=0
		N =30
		for row in readCSV:

			n = len(row)
			# print n
			current=0
			count2=0
			count=0
			if(row[0]=='NaN' or row[0] == ''):
				while( current < n and (row[current] =='NaN' or row[current].strip() =='') ):
					# print current
					current+=1

			if(current == n):
				continue

			if(current < n  and (row[current].strip()!='' or row[current]!='NaN')):
				temp = row[current]
	
			temp.strip()
			# if(rowcount==29):
			# 	print row
			
			for i in range(0,n):
				# print count
				
				if(row[i] == 'NaN' or row[i].strip() ==''):
					count2+=1		
					row[i]=temp
				else:
					temp = row[i]
				count+=1

			
			# print n,N
			if n < N:
				addList = [row[n-1]]*(N-n)
				# print addList
				row.extend(addList)
			else:
				row = row[:30]

			# print len(row)
			if(count2<25):
				rows.append(row)
			rowcount+=1
			
			# print row[n-1]

	with open(fname+'_updated.csv',mode='w') as file:
		writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		# writer.writerow(header)
		for row in rows:
			writer.writerow(row)
	

# 2,11,26,74 deleted. form 3rd person