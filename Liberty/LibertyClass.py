import numpy as np
from CleanDataClass import CleanDataHelper
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import sys
import random


class LibertyClass:
	cleanDataHelper = CleanDataHelper()
	Xtrain = 0
	Xtest = 0
	Ytrain = 0
	ids = []

	def __init__(self, filename, testFilename):
		(self.Xtrain, self.Ytrain, self.ids) = self.cleanDataHelper.getDataCsv(filename)
		(self.Xtest, Ytest, self.ids) = self.cleanDataHelper.getDataCsv(testFilename, yValue = True, shuffle = False)
		
	def trainSVM(self):
		clf = svm.SVR(kernel="linear")
		clf.fit(self.Xtrain, self.Ytrain)
		return clf

	def trainSGD(self):
		clf = SGDRegressor(loss="squared_loss",n_iter = np.ceil(10**3))
		clf.fit(self.Xtrain, self.Ytrain)
		return clf

	def printValidation(self, clf):
		print('Id,Hazard')
		for i, row in enumerate(self.Xtest):
			print(str(self.ids[i]) + ","+str(clf.predict(row)[0]))

