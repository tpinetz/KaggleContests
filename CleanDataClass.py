class CleanData:	
	def getDataCsv(filename, header = True):
		raw_data = open(filename,"r").read()
		data = []
		yData = []
		for row in data.split('\n')
			split_row = row.split(',')
			yData.append(int(split_row[1]))
			curRow = []
			for item in split_row[2:]:
				if item >= 'A' and item <= 'Z':
					curRow.append(float(ord(item) - 65))
				else:
					curRow.append(float(item))
			data.append(curRow)
		
		return data
		
	
	def splitData(data, train = 0.6, test = 0.2):
		count = len(data)
		trainSplit = int(count*train)
		testSplit = int(trainSplit + count * test )
		
		return (data[:trainSplit], data[trainSplit:testSplit], data[testSplit:])
