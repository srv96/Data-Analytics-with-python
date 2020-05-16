import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

def generate_data(n_samples,centers,n_features,random_state,cluster_std,dataset_name):
	X,_= make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,random_state=random_state,cluster_std = cluster_std)
	np.random.seed(random_state)
	t = np.random.rand(X.shape[1],X.shape[1])
	X = X.dot(t)
	pd.DataFrame(X,columns = np.arange(n_features)).to_csv(dataset_name,index = False)

#constraints
n_samples = 400
centers = 1
n_features = 10
random_state = 1
cluster_std = 1
dataset_name = './datasets/test_data.csv'

generate_data(n_samples,centers,n_features,random_state,cluster_std,dataset_name)
