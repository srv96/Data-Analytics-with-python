import numpy as np

def train_test_split(X,y,test_ratio = 0.25,shuffle = True):
	data = np.hstack((X,y))
	size = data.shape[0]
	train_ratio =  1 - test_ratio
	train_size,test_size = int(train_ratio * size) , int(test_ratio * size)

	if train_size + test_size < size :
		train_size = train_size + 1

	flags = np.concatenate((np.full(train_size,True),np.full(test_size,False)))

	if shuffle:
		np.random.shuffle(flags)
	
	train_data = data[np.where(flags)]
	test_data = data[np.where(np.logical_not(flags))]

	return train_data[:,:-1],train_data[:,-1],test_data,test_data





