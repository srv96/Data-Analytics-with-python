import numpy as np
import itertools
import pydot

class FPGrowth :
	def __init__(self):
		pass
	# node skeleton
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

	#insert every tupple into the fp-tree
	def insert_tupple(self,root,tup):
		for ele in tup:
			if ele in root.childs_name:
				root.childs[ele].freq = root.childs[ele].freq + 1
				root = root.childs[ele]
			else:
				root.childs_name.append(ele)
				root.childs[ele] = self.Node(ele)
				root = root.childs[ele]

	#process raw data
	def set_priority_list(self):
		item_aval = list(set(itertools.chain.from_iterable(self.raw_data)))
		item_aval.sort()
		table = {}
		for item in item_aval:
			freq = 0
			for tup in self.raw_data:
				for ele in tup:
					if ele == item:
						freq = freq + 1
			if freq >= self.support:
				table[item] = freq
		table = sorted(table.items(),key = lambda kv:(kv[1], kv[0]),reverse = True)
		length = len(table)
		self.priority_list = []
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
			self.priority_list.append(similar)
		self.priority_list = list(itertools.chain.from_iterable(self.priority_list))
		return self.priority_list

	#filter the data according to the priority	
	def set_filter_data(self):
		self.filter_data = []
		for tup in self.raw_data:
			filter_tup = []
			for item in self.priority_list:
				if item in tup:
					filter_tup.append(item)
			if len(filter_tup) > 0:
				self.filter_data.append(filter_tup)

	# make tree from prioritylist and filter data
	def make_tree(self,raw_data,support):
		self.raw_data,self.support = raw_data,support
		self.set_priority_list()
		self.set_filter_data()
		self.root = self.Node('root')
		for tup in self.filter_data:
			curr = self.root
			self.insert_tupple(curr,tup)

	#tree visualization in the console
	def visualize(self):
		self.root.disp(0)


	#graphical representation of tree
	def add_sub_tree(self,root):
		if root == None :
			return
		sub_tree = {}
		children = root.childs_name
		for child in children:
			sub_tree[child +'[id:'+str(self.counter+1)+']  ' + str(root.childs[child].freq)] = self.add_sub_tree(root.childs[child])
			self.counter+=1
		return sub_tree

	def make_dict(self):
		self.counter = 0
		info_dict = {}
		info_dict[self.root.item + ' ' + str(self.root.freq)] = self.add_sub_tree(self.root)
		return info_dict

	def draw(self,parent_name, child_name):
	    edge = pydot.Edge(parent_name, child_name)
	    self.graph.add_edge(edge)

	def visit(self,node, parent=None):
	    for k,v in node.items():
	        if isinstance(v, dict):
	            if parent:
	                self.draw(parent, k)
	            self.visit(v, k)
	        else:
	            self.draw(parent, k)
	            self.draw(k, k+'_'+v)

	def save_to_dir(self,filename):
		self.info_dict = self.make_dict()
		self.graph = pydot.Dot(graph_type='graph')
		self.visit(self.info_dict)
		self.graph.write_png(filename)


#load data from file
def load_data(filename) :
	raw_data = open(filename)
	pro_data = []
	for data in raw_data:
		tup = []
		for ele in data:
			if ele!='\n':
				tup.append(ele)
		tup.sort()
		pro_data.append(tup)
	return pro_data


support = 20
filename = 'result.png'
raw_data = load_data('dataset.txt')
refl = FPGrowth()
refl.make_tree(raw_data,support)
#refl.visualize()
refl.save_to_dir(filename = filename)