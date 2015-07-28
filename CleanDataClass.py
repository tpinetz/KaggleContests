import numpy as np

class CleanDataHelper:	
	def getDataCsv(self, filename, header = True, yValue= False):
		raw_data = open(filename,"r").read()
		data = []
		yData = []
		ids = []

		offset = (1 if header else 0)      
		
		for row in raw_data.split('\n')[offset:]:
			split_row = row.split(',')
			if len(split_row) < 2:
				continue
			yData.append(int(split_row[1]))
			ids.append(int(split_row[0]))
			curRow = []
			for item in split_row[(2-yValue):]:
				if item >= 'A' and item <= 'Z':
					curRow.append(float(ord(item) - 65))
				else:
					curRow.append(float(item))
			data.append(curRow)
		
		return (data, yData, ids)
		
	def normalizeData(self, data):
		count = len(data)
		mins = [999999 for i in range(0,len(data[0]))]
		maxs = [-99999 for i in range(0,len(data[0]))]
		
		for row in data:
			for i, item in enumerate(row):
				if item < mins[i]:
					mins[i] = item
				if item > maxs[i]:
					maxs[i] = item 
		
		result = []
		for row in data:
			split_result = []
			for i, item in enumerate(row):
				split_result.append(((item - mins[i])/(maxs[i]-mins[i]))*2-1)
			result.append(split_result)
		
		return result
	
	def splitData(self, data, train = 0.6, test = 0.2):
		count = len(data)
		trainSplit = int(count*train)
		testSplit = int(trainSplit + count * test )
		
		return (np.array(data[:trainSplit]), np.array(data[trainSplit:testSplit]), np.array(data[testSplit:]))

	def validate(self, classifier, xdata, ydata):
		count = len(xdata)
		cor = 0
		dis=0
		
		for i, row in enumerate(xdata):
			pred = classifier.predict(row)

			if pred == ydata[i]:
				cor += 1
			else:
				dis += (pred-ydata[i])**2
				
		print("Number correct: " + str(cor))
		print("Number incorrect: " + str(count - cor))
		print("Mean squared distance: " + str(dis/count))
		
