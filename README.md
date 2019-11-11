# CSE-572-Data-Mining
For ASG 2
Training:
Run python replacemissingvalues.py filename  for all the testFiles individually ( it will use the first 30 columns for extracting features )
eg: python replacemissingvalues.py Data/Nomeal5.csv. It will create files with _updated suffix
We have added the label column manually. 1 is the value of 31st column for meal files and 0 is the value of 31st column for no meal files.
Then run the concatenateAndShuffleData.py  file . It will concatenate all the files with _updated suffix by shuffling all the rows.

Models:
1. Random Forest
2. Decision Tree
3. Neural Network
4. Support Vector Machine

Model Training:
For training the models, "allModelTraining.py" needs to be run. We have already trained the models and the models are stored in .sav files which will be used ofr testing on the models.
The models can be trained:
    python allModelTraining.py

Testing:
For testing all the classifier models, test.py needs to be run:
      python test.py sampletest.csv
Model training is done with 64 bit python, so if 32 bit python is used for testing, the model needs to be trained before testing.

