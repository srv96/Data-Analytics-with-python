import numpy as np
import matplotlib.pyplot as plt

def expand_feature(X,feature):
		return np.hstack(tuple([np.power(X,i) for i in range(feature)]))

def generate_data(θ,limit,size,noise,dataset_name):
	noise = np.random.normal(0,noise,size).reshape(size,1)
	X = (np.random.rand(size)*(limit[1]-limit[0]) + limit[0]).reshape(size,1) + noise
	y = expand_feature(X,θ.size).dot(θ.T).flatten()
	X = X.flatten()

	file = open(dataset_name,'w+')
	for i in range(len(X)):
		file.write(str(X[i]))
		file.write(" ")
		file.write(str(y[i]))
		file.write('\n')
	file.close()

#constraints
noise = 9
limit = 9,10
size = 10
dataset_name = "./test_dataset.txt"
θ = np.array([[-10,-10,4.1,10,-0.4]])

generate_data(θ,limit,size,noise,dataset_name)


