trainx = []
trainy = []

mean = []
maxV = []

rows = open('train.csv','r').read().split('\n')
	
rows = rows[:-1]
for row in rows[1:]:
	split_row = row.split(',')
	result = []
	for item in split_row:
		if item >= 'A' and item <= 'Z':
			result.append(ord(item) - 65)
		elif item != '':
			result.append(int(item))
	trainx.append([result[0]] + result[2:])
	trainy.append([result[1]])
	
for row in trainx:
	for item in row:
		print (str(item) + " "),
	print ""
