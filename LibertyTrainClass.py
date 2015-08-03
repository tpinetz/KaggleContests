import numpy as np
from CleanDataClass import CleanDataHelper
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import sys
import random


class LibertyTrainClass:
	cleanDataHelper = CleanDataHelper()
	Xtrain = 0
	Xtest = 0
	Ytrain = 0
	Ytest = 0
	ids = []

	def __init__(self, filename):
		(X, Y, self.ids) = self.cleanDataHelper.getDataCsv(filename)
		(self.Xtrain, self.Xtest) = self.cleanDataHelper.splitData(X)
		(self.Ytrain, self.Ytest) = self.cleanDataHelper.splitData(Y)
		(self.Xtrain, self.Xtest) = self.cleanDataHelper.normalizeWithMinMaxScaler(self.Xtrain, self.Xtest)

	def trainSVM(self):
		clf = svm.SVR()
		clf.fit(self.Xtrain, self.Ytrain)
		return clf

	def trainSGD(self):
		clf = SGDRegressor(loss="squared_loss",n_iter = np.ceil(10**3))
		clf.fit(self.Xtrain, self.Ytrain)
		return clf

	def printValidation(self, clf):
		self.cleanDataHelper.validate(clf, self.Xtest, self.Ytest)

	def printData(self, amount):
		print("ids:")
		print(self.ids[:amount])
		print("data:")
		print(self.Xtrain[:amount])
		
