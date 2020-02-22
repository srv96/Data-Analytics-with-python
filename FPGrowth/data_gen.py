import numpy as np

#create dataset and store it
max_transaction = 500
max_items = 500
data = []
n1 = int(np.random.rand()*max_transaction)
for i in range(n1):
	row = []
	n2 = int(np.random.rand()*max_items)
	for j in range(n2):
		ch = chr(ord('A')+int(np.random.rand()*26))
		row.append(ch)
	data.append(row)

file = open('dataset.txt','w')
for row in data:
	for ele in row:
		file.write(ele)
	file.write('\n')
file.close()