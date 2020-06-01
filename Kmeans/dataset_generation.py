import numpy as np
from sklearn.datasets import make_blobs


def generate_data(n_samples,centers,n_features,random_state,cluster_std,dataset_name):
	X,_= make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,random_state=random_state,cluster_std = cluster_std)
	np.savetxt("test_dataset.txt",X, delimiter=",")

n_samples = 400
centers = 10
n_features = 2
random_state = 1
cluster_std = 0.25
dataset_name = './test_dataset.txt'

generate_data(n_samples,centers,n_features,random_state,cluster_std,dataset_name)