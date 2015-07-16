import sys

arguments = sys.argv

rows = open(arguments[1], "r").read().split('\n')

count = len(rows)
trainSplit = int(count*0.6)
testSplit = int(trainSplit + count*0.2 )

train = open(arguments[1] + ".train", "w+")
test = open(arguments[1] + ".test","w+")
val = open(arguments[1] + ".val","w+")


for i in range(0, trainSplit):
	train.write(rows[i] + "\n")
	
for i in range(trainSplit,testSplit):
	test.write(rows[i] + "\n")

for i in range(testSplit,count):
	val.write(rows[i] + "\n")
