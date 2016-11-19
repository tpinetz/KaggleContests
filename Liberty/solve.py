import numpy as np
from sklearn import linear_model
import sys

def readFromFileX(curFile):
	res = []
	for row in curFile.read().split('\n'):
		result = []
		for item in row.split("  ")[1:]:
			if item == '': 
				continue
			result.append(int(item))
	
		res.append(result)
	return res

def readFromFileY(curFile):
	res = []
	for row in curFile.read().split('\n'):
		if row == '':
			continue
		res.append(int(row))
	return res


trainFileX = open(sys.argv[1] + ".x.train","r")
testFileX = open(sys.argv[1] + ".x.test","r")

trainFileY = open(sys.argv[1] + ".y.train","r")
testFileY = open(sys.argv[1] + ".y.test","r")

trainX = readFromFileX(trainFileX)
trainY = readFromFileY(trainFileY)

testX = readFromFileX(testFileX)
testY = readFromFileY(testFileY)


clf = linear_model.SGDClassifier()


ntrainX = np.array(trainX[:-1])
ntrainY = np.array(trainY)

print ntrainX.shape
print ntrainY.shape

clf.fit(ntrainX,ntrainY)

count = len(testX)
correct = 0

for i, row in enumerate(testX):
	if len(row) == 0:
		continue
	ytemp = clf.predict([row])
	if ytemp == testY[i]:
		correct +=1
	else:
		print "prediction: " + str(ytemp[0]) + " / real Value: " + str(testY[i])
	
print "correct: " + str(correct) + "/ failed: " + str((count - correct))

