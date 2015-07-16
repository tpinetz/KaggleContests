class CleanData:

	def __init__():
		self.data = []
	
	def splitData(filename):
		rows = open(filename, "r").read().split('\n')

		count = len(rows)
		trainSplit = int(count*0.6)
		testSplit = int(trainSplit + count*0.2 )

		train = open(filename + ".train", "w+")
		test = open(filename + ".test","w+")
		val = open(filename + ".val","w+")


		for i in range(0, trainSplit):
			train.write(rows[i] + "\n")
	
		for i in range(trainSplit,testSplit):
			test.write(rows[i] + "\n")

		for i in range(testSplit,count):
			val.write(rows[i] + "\n")
