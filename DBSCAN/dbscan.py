from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import itertools

class dbscan:
	def __init__(self):
		pass
		
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

	def distance(self,p1,p2):
		return np.sqrt(np.sum(np.square(p1-p2),axis = 0))

	def check_core(self,p_point,container,ε,minPts):
		dist = {}
		for s_point in container:
			dist[s_point] = self.distance(p_point.get_loc(),s_point.get_loc())
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

	def check_border(self,p_point,container,ε,minPts):
		if p_point.get_status() == 'core' :
			return 'core'
		else :
			dist = {}
			for s_point in container:
				dist[s_point] = self.distance(p_point.get_loc(),s_point.get_loc())
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

	def cluster_dbscan(self,X,ε,minPts):
		self.container = clf.make_datapoints(X)
		for point in self.container:
			point.set_status(self.check_core(point,self.container,ε,minPts))
		for point in self.container:
			point.set_status(self.check_border(point,self.container,ε,minPts))

	#data pre-processing
	def make_datapoints(self,X):
		container = []
		for tup in X:
			container.append(self.point(tup))
		return container


	#data visualization
	def plot_cluster(self):
		core = []
		border = []
		noise = []

		for point in self.container:
			if point.get_status() == 'core':
				core.append(point.get_loc())
			if point.get_status() == 'border':
				border.append(point.get_loc())
			if point.get_status() == 'noise':
				noise.append(point.get_loc())

		core = np.array(core)
		border = np.array(border)
		noise = np.array(noise)

		n_feature = self.container[0].get_loc().size
		
		if n_feature == 2:
			if len(core) != 0: 
				plt.scatter(core[:,0],core[:,1],color = 'red',label = "core",s=8)
			if len(border) != 0:
				plt.scatter(border[:,0],border[:,1],color = 'blue',label = "border",s=4)
			if len(noise) != 0:
				plt.scatter(noise[:,0],noise[:,1],color = 'green',label = "noise",s=2)
			plt.gcf().canvas.set_window_title("DBSCAN clustering")
			plt.legend(loc='upper right')
			plt.show()

		elif n_feature == 3:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			if len(core) != 0: 
				ax.scatter(core[:,0],core[:,1],core[:,2],color = 'red',label = "core",s=8)
			if len(border) != 0:
				ax.scatter(border[:,0],border[:,1],border[:,2],color = 'blue',label = "border",s=4)
			if len(noise) != 0:
				ax.scatter(noise[:,0],noise[:,1],noise[:,2],color = 'green',label = "noise",s=2)
			plt.gcf().canvas.set_window_title("DBSCAN clustering")
			plt.legend(loc='upper right')
			plt.show()


#fetch data from the file
def get_data_from_file(filename):
    datafile = open(filename)
    data = []
    for row in datafile:
        tup = []
        for ele in row.split(','):
            tup.append(float(ele))
        data.append(np.array(tup))
    return np.array(data)
filename = 'test_dataset.txt'
ε = 0.50
minPts = 5
X = get_data_from_file(filename = filename)



#classify data
clf = dbscan()
clf.cluster_dbscan(X,ε,minPts)
clf.plot_cluster()