# CSE-572-Data-Mining
For ASG 2
Training:
Run python replacemissingvalues.py filename  for all the testFiles individually ( it will use the first 30 columns for extracting features )
eg: python replacemissingvalues.py Data/Nomeal5.csv. It will create files with _updated suffix
We have added the label column manually. 1 is the value of 31st column for meal files and 0 is the value of 31st column for no meal files.
Then run the concatenateAndShuffleData.py  file . It will concatenate all the files with _updated suffix by shuffling all the rows.
