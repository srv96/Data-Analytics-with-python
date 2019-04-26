from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import itertools

def distance(p1,p2):
	return np.sqrt(np.sum(np.square(p1-p2),axis = 1))

def factors(n):    
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def get_middle(n):
	facts = list(factors(n))
	if np.sqrt(n) - int(np.sqrt(n)) == 0:
		return [int(np.sqrt(n)),int(np.sqrt(n))]
	else:
		facts.sort()
		return [facts[int(len(facts)/2)-1],facts[int(len(facts)/2)]]
	return facts

def plot_cluster(result):
	n_features = len(result[0][0])
	if n_features == 2 : 
		for cluster in result:
			plt.scatter(cluster[:,0],cluster[:,1])
		plt.show()
	if n_features == 3 :
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for cluster in result:
			ax.scatter(cluster[:,0],cluster[:,1],cluster[:,2])
		plt.show()
	else:
		row,col = get_middle((n_features * (n_features - 1)) / 2)
		fig, ax = plt.subplots(nrows=int(row), ncols=int(col))
		pairs = list(itertools.combinations(range(n_features),2))
		inc = 0
		for i in range(len(ax)):
			for j in range(len(ax[0])):
				for k in range(len(result)):
					ax[i][j].scatter(result[k][:,pairs[inc][0]],result[k][:,pairs[inc][1]],s=0.2)
				inc = inc + 1
		plt.show()

n_samples = 1000
centers = 3
n_features = 4
epoch = 10
X,y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,random_state=0,cluster_std = 0.5)
d_points = np.array(X)
C = d_points[:centers]
d_points = d_points[centers:]

dists = np.zeros((centers,d_points.shape[0]))
for i,c in enumerate(C):
	for j,points in enumerate(d_points):
		dists[i][j] = distance(c,points)

classify = np.zeros(d_points.shape[0])
for i in range(d_points.shape[0]):
	temp = dists[:,i].ravel()
	classify[i] = np.where(temp == np.amin(temp))[0]
n_c = np.zeros((centers,n_features))
classify =classify.astype(int)
count = np.bincount(classify)
count= count.reshape(len(count),1)
for i in range(d_points.shape[0]):
	n_c[int(classify[i]),:] = n_c[int(classify[i]),:] + d_points[i,:]
n_c = n_c / count
d_points = np.append(d_points,C,axis = 0)

for e in range(epoch):
	C = n_c
	dists = np.zeros((centers,d_points.shape[0]))
	for i,c in enumerate(C):
		for j,points in enumerate(d_points):
			dists[i][j] = distance(c,points)

	classify = np.zeros(d_points.shape[0])
	for i in range(d_points.shape[0]):
		temp = dists[:,i].ravel()
		classify[i] = np.where(temp == np.amin(temp))[0]

	n_c = np.zeros((centers,n_features))
	classify =classify.astype(int)
	count = np.bincount(classify)
	count= count.reshape(len(count),1)
	for i in range(d_points.shape[0]):
		n_c[int(classify[i]),:] = n_c[int(classify[i]),:] + d_points[i,:]
	n_c = n_c / count

classify = np.zeros(d_points.shape[0])
for i in range(d_points.shape[0]):
	temp = dists[:,i].ravel()
	classify[i] = np.where(temp == np.amin(temp))[0]
classify = classify.astype(int)

result = []
for i in range(centers):
	result.append([])
for i in range(d_points.shape[0]):
	result[classify[i]].append(list(d_points[i]))
for i in range(centers):
	result[i] = np.array(result[i])


result = np.array(result)

plot_cluster(result)