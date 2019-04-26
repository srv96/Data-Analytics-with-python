import numpy as np
import itertools
def process_data(dataset,support):
	raw_data = open(dataset)
	pro_data = []
	for data in raw_data:
		tup = []
		for ele in data:
			if ele!='\n':
				tup.append(ele)
		tup.sort()
		pro_data.append(tup)
	item_aval = list(set(itertools.chain.from_iterable(pro_data)))
	item_aval.sort()
	table = {}
	for item in item_aval:
		freq = 0
		for tup in pro_data:
			for ele in tup:
				if ele == item:
					freq = freq + 1
		if freq >= support:
			table[item] = freq
	table = sorted(table.items(),key = lambda kv:(kv[1], kv[0]),reverse = True)
	length = len(table)
	pattern = []
	i = 0
	while i < length:
		j = 0
		similar = []
		while j < length-i:
			if table[i][1] == table[i+j][1]:
				similar.append(table[i+j][0])
			else:
				break
			j = j + 1
		i = i + j
		similar.sort()
		pattern.append(similar)
	pattern = list(itertools.chain.from_iterable(pattern))
	filter_data = []
	for tup in pro_data:
		filter_tup = []
		for item in pattern:
			if item in tup:
				filter_tup.append(item)
		if len(filter_tup) > 0:
			filter_data.append(filter_tup)
	raw_data.close()
	return filter_data

def insert_tupple(root,tup):
	for ele in tup:
		if ele in root.childs_name:
			root.childs[ele].freq = root.childs[ele].freq + 1
			root = root.childs[ele]
		else:
			root.childs_name.append(ele)
			root.childs[ele] = Node(ele)
			root = root.childs[ele]
			
class Node :
	def __init__(self,item):
		self.item = item
		self.freq = 1
		self.childs_name= []
		self.childs = {}
	def disp(self,height):
		print(('     '*height),self.item,'  ',self.freq)
		for child in self.childs.keys():
			self.childs[child].disp(height + 1)



pro_data = process_data('minidataset.txt',3)

root = Node('root')
for tup in pro_data:
	curr = root
	insert_tupple(curr,tup)
root.disp(0)