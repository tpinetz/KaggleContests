import numpy as np
from CleanDataClass import CleanDataHelper
from sklearn import linear_model
from sklearn import svm
import sys
import random

    
helper = CleanDataHelper()

data = []
ydata = []
ids = []

(data,ydata, ids) = CleanDataHelper.getDataCsv(sys.argv[1])
(dataTest, ydataTest, ids) = CleanDataHelper.getDataCsv(sys.argv[2], yValue=True)

data = CleanDataHelper.normalizeData(data)

random.shuffle(data)

(trainX, testX, valX) = CleanDataHelper.splitData(data,1.,0.0)
(trainY, testY, valY) = CleanDataHelper.splitData(ydata,1.,0.0)

#print(trainX.shape)
#print(trainY.shape)

#### Stochastic Gradient Descent ####
#### Hard underfitting ####

#clf = linear_model.SGDClassifier(n_iter=100, loss='log', alpha = 0.01)
#clf.fit(trainX,trainY)

#### Support vector Machine ####

#print("Start Training!")

clf = svm.SVC(gamma=0.001, C = 100)
clf.fit(trainX,trainY)

#print("Start validating")

#CleanDataHelper.validate(clf,testX,testY)
#CleanDataHelper.validate(clf,testX,testY)

print('Id,Hazard')

for i, row in enumerate(dataTest):
	print(str(ids[i]) + ","+str(int(round(clf.predict(row)[0]))))

