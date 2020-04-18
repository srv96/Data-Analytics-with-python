import numpy as np
import pandas as pd

def expand_feature(X,feature):
		return np.hstack(tuple([np.power(X,i) for i in range(feature)]))

def generate_data(θ,limit,size,noise,dataset_name):
	noise = np.random.normal(0,noise,size).reshape(size,1)
	X = (np.random.rand(size)*(limit[1]-limit[0]) + limit[0]).reshape(size,1) + noise
	y = expand_feature(X,θ.size).dot(θ.T)
	pd.DataFrame(np.hstack((X,y)),columns = ['X','y']).to_csv(dataset_name,index = False)


#constraints
noise = 1
limit = 0,10
size = 1000
dataset_name = "./datasets/test_data.csv"
θ = np.array([[-10,-10,4.1,10,-0.4]])

generate_data(θ,limit,size,noise,dataset_name)


