import numpy as np
from CleanDataClass import CleanDataHelper
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import sys
import random

    
helper = CleanDataHelper()

data = []
ydata = []
ids = []


#### Support vector Machine ####
def trainSVM(trainX,trainY):
	clf = svm.SVC(gamma=0.001, c = 100)
	clf.fit(trainX, trainY)
	return clf

#### Stochastic Gradient Descent ####
#### Hard underfitting ####
def trainSGD(trainX, trainY):
	clf = SGDRegressor(loss="squared_loss",n_iter = np.ceil(10**6))
	clf.fit(trainX, trainY)
	return clf	


(data,ydata, ids) = helper.getDataCsv(sys.argv[1])

if len(sys.argv) > 2:
	(dataTest, ydataTest, ids) = helper.getDataCsv(sys.argv[2], yValue=True)
	
	
#data = helper.normalizeData(data)

random.shuffle(data)


if len(sys.argv) > 2:
	(trainX, testX, valX) = helper.splitData(data,1.,0.)
	(trainY, testY, valY) = helper.splitData(ydata,1.,0.)
else:
	(trainX, testX, valX) = helper.splitData(data,0.5,0.3)
	(trainY, testY, valY) = helper.splitData(ydata,0.5,0.3)


scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)


if len(sys.argv) < 3:
	print("Start Training!")
	testX = scaler.transform(testX)

clf = trainSGD(trainX, trainY)


if len(sys.argv) < 3:
	print(clf)
	print("Start validating")
	helper.validate(clf,testX,testY)


if len(sys.argv) > 2:
	dataTest = scaler.transform(dataTest)
	print('Id,Hazard')
	for i, row in enumerate(dataTest):
		print(str(ids[i]) + ","+str(clf.predict(row)[0]))
