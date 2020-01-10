from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import itertools

class point:
	def __init__(self,loc):
		self.loc = loc
		self.status = "unknown"
	def get_loc(self):
		return self.loc
	def set_status(self,status):
		self.status = status
	def get_status(self):
		return self.status

def distance(p1,p2):
	return np.sqrt(np.sum(np.square(p1-p2),axis = 0))

def check_core(p_point,container,ε,minPts):
	dist = {}
	for s_point in container:
		dist[s_point] = distance(p_point.get_loc(),s_point.get_loc())
	dist = sorted(dist.items(),key = lambda kv:(kv[1], kv[0]))[1:minPts+1]
	for i in range(len(dist)):
		dist[i] = list(dist[i]) 
	i = 0
	while i < len(dist) :
		if dist[i][1] > ε:
			del dist[i]
			i = i - 1
		i = i + 1
	if len(dist) == minPts:
		return "core"
	else :
		return"unknown"

def check_border(p_point,container,ε,minPts):
	if p_point.get_status() == 'core' :
		return 'core'
	else :
		dist = {}
		for s_point in container:
			dist[s_point] = distance(p_point.get_loc(),s_point.get_loc())
		dist = sorted(dist.items(),key = lambda kv:(kv[1], kv[0]))[1:minPts+1]
		for i in range(len(dist)):
			dist[i] = list(dist[i]) 
		i = 0
		while i < len(dist) :
			if dist[i][1] > ε:
				del dist[i]
				i = i - 1
			i = i + 1
		if len(dist) < minPts:
			for point in dist:
				if point[0].get_status() == 'core':
					return "border"
		return "noise"

def cluster_dbscan(container,ε,minPts):
	for point in container:
		point.set_status(check_core(point,container,ε,minPts))
	for point in container:
		point.set_status(check_border(point,container,ε,minPts))

def make_datapoints(X):
	container = []
	for tup in X:
		container.append(point(tup))
	return container

def plot_cluster(container):
	core = []
	border = []
	noise = []

	for point in container:
		if point.get_status() == 'core':
			core.append(point.get_loc())
		if point.get_status() == 'border':
			border.append(point.get_loc())
		if point.get_status() == 'noise':
			noise.append(point.get_loc())

	core = np.array(core)
	border = np.array(border)
	noise = np.array(noise)
	
	if len(core) != 0: 
		plt.scatter(core[:,0],core[:,1],color = 'red',label = "core",s=8)
	if len(border) != 0:
		plt.scatter(border[:,0],border[:,1],color = 'blue',label = "border",s=4)
	if len(noise) != 0:
		plt.scatter(noise[:,0],noise[:,1],color = 'green',label = "noise",s=2)
	plt.gcf().canvas.set_window_title("DBSCAN clustering")
	plt.legend(loc='upper right')
	plt.show()



n_samples = 100
centers = 3
n_features = 2
ε = 0.25
minPts = 10
X,_= make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,random_state=0,cluster_std = 0.2)
container = make_datapoints(X)
cluster_dbscan(container,ε,minPts)
plot_cluster(container)

