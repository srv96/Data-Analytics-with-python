import numpy as np
import itertools

#filtering technique
def find_large_support(table,support_count):
	neglect = []
	for frequent_itemset in table:
		if table[frequent_itemset] < support_count :
			neglect.append(frequent_itemset)
	for frequent_itemset in neglect:
		del table[frequent_itemset]
	return table

#hash_function for selecting bucket for itemset 
def get_hash_value(key1,key2):
	return ((ord(key1)-64)*10 + (ord(key2)-64)) % 7

#improved filtering technique using hashing only for C2
def find_large_support2(table,raw_data,support_count):
	buckets = {}
	for i in range(7):
		buckets[i] = {}
	keys = list(table.keys())
	size = len(keys)
	for i in range(size-1):
		for j in range(i+1,size):
			frequent_itemset = keys[i] + keys[j]
			bucket_num = get_hash_value(keys[i],keys[j])
			buckets[bucket_num][frequent_itemset] = find_frequency_count(raw_data,frequent_itemset)
	for i in range(7):
		count = 0
		for frequent_itemset in buckets[i]:
			count = count + buckets[i][frequent_itemset]
		if count < support_count:
			del buckets[i]
	candidate_table = {}
	for bucket in buckets:
		for frequent_itemset in buckets[bucket]:
			candidate_table[frequent_itemset] = buckets[bucket][frequent_itemset] 
	return candidate_table

#purning technique
def is_valid_association(table,key):
	keys = table.keys()
	comb = list(itertools.combinations(key,len(key)-1))
	com = []
	for tup in comb:
		com.append(''.join(tup))
	for ele in com:
		if ele not in keys:
			return False
	return True

#filter the table after purning
def filter_valid_association(table,key_list):
	filter_list = []
	for key in key_list:
		if is_valid_association(table,key):
			filter_list.append(key)
	return filter_list

#making higher association
def is_similar(key1,key2):
	if key1[:-1] == key2[:-1] :
		return True
	else :
		return False

#find higher association
def get_higher_association(table):
	h_assos = []
	keys = list(table.keys())
	i = 0
	while i < len(keys)-1 :
		j = i+1
		while j < len(keys):
			if is_similar(keys[i],keys[j]):
				key = keys[i]+keys[j][-1]
				key = ''.join(sorted(key))
				h_assos.append(key)
			j = j + 1
		i = i + 1
	return h_assos

#finding the frequency count of each itemset in the raw_data
def find_frequency_count(raw_data,frequent_itemset):
	freq = 0
	for lst in raw_data:
		count = 0
		for ele in frequent_itemset:
			if ele in lst :
				count = count + 1
		if count == len(frequent_itemset):
			freq = freq + 1
	return freq

#get candidate table by doing frequency count of all itemset 
def get_cadidate_key_table(raw_data,frequent_itemset_list):
	C = {}
	for frequent_itemset in frequent_itemset_list:
		C[frequent_itemset] = find_frequency_count(raw_data,frequent_itemset)
	return C

#get candidate tables	
def find_association_rules(raw_data,support):
	#finding all the sold item set
	item_found = ''
	for st in raw_data:
		for ch in st:
			item_found+=ch
	item_found = list(set(item_found))
	item_found.sort()
	Candidate_set = {}
	rule_making_possible = True
	i = 1
	while rule_making_possible :
		print("Generating candidate table :==> " ,i)
		if i == 1 :
			Candidate_set[i] = get_cadidate_key_table(raw_data,item_found)
			Candidate_set[i] = find_large_support(Candidate_set[i],support)
		elif i == 2:
			higher_assos = get_higher_association(Candidate_set[i-1])
			Candidate_set[i] = find_large_support2(Candidate_set[i-1],raw_data,support)
			Candidate_set[i] = find_large_support(Candidate_set[i],support)
		else:
			if len(Candidate_set[i-1]) < 1:
				rule_making_possible = False
				break 
			higher_assos = get_higher_association(Candidate_set[i-1])
			higher_assos = filter_valid_association(Candidate_set[i-1],higher_assos)
			Candidate_set[i] = get_cadidate_key_table(raw_data,higher_assos)
			Candidate_set[i] = find_large_support(Candidate_set[i],support)
		i = i + 1
	return Candidate_set

#importing data
file = open("dataset.txt",'r')
raw_data =file.read()
file.close()

#data preprocessing
raw_data = raw_data.split('\n')
for i in range(len(raw_data)):
	raw_data[i] = sorted(set(raw_data[i]))

#apriori algorithm
support = 2
Candidate_set = find_association_rules(raw_data,support)
print(Candidate_set)
