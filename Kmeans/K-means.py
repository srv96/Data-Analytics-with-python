from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import itertools

class Kmeans:
	def __init__(self,init = None):
		self.init = init

	def distance(self,p1,p2):
		return np.sqrt(np.sum(np.square(p1-p2),axis = 0))

	def factors(self,n):
		return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

	def initialize_centroids(self):
		centroids = np.zeros([self.n_clusters,self.X.shape[1]])
		centroids[0] = self.X[np.random.randint(self.X.shape[0]),:]
		for i in range(self.n_clusters):
			all_distances = np.array([])
			for data_point in self.X:
				near_centroids_distance = np.linalg.norm(data_point-centroids,axis=1).min()
				all_distances = np.append(all_distances,near_centroids_distance)
			centroids[i]= self.X[np.argmax(all_distances),:]
		return centroids

	def get_middle(self,n):
		facts = list(self.factors(n))
		if np.sqrt(n) - int(np.sqrt(n)) == 0:
			return [int(np.sqrt(n)),int(np.sqrt(n))]
		else:
			facts.sort()
			return [facts[int(len(facts)/2)-1],facts[int(len(facts)/2)]]
		return facts

	def visualize(self):
		n_features = len(self.prediction[0][0])
		if n_features == 2 : 
			for cluster in self.prediction:
				plt.scatter(cluster[:,0],cluster[:,1])
			plt.show()
			return
		elif n_features == 3 :
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			for cluster in self.prediction:
				ax.scatter(cluster[:,0],cluster[:,1],cluster[:,2])
			plt.show()
			return
		else:
			row,col = self.get_middle((n_features * (n_features - 1)) / 2)
			fig, ax = plt.subplots(nrows=int(row), ncols=int(col))
			pairs = list(itertools.combinations(range(n_features),2))
			inc = 0
			for i in range(len(ax)):
				for j in range(len(ax[0])):
					for k in range(len(self.prediction)):
						ax[i][j].scatter(prediction[k][:,pairs[inc][0]],self.prediction[k][:,pairs[inc][1]],s=0.2)
					inc = inc + 1
			plt.show()
			return

	def set_center(self):
		self.dists = np.zeros((self.n_clusters,self.X.shape[0]))
		for i,c in enumerate(self.C):
			for j,points in enumerate(self.X):
				self.dists[i][j] = self.distance(c,points)
		classify = np.zeros(self.X.shape[0])
		for i in range(self.X.shape[0]):
			temp = self.dists[:,i].ravel()
			classify[i] = np.where(temp == np.amin(temp))[0]
		self.n_c = np.zeros((self.n_clusters,self.X.shape[1]))
		classify =classify.astype(int)
		count = np.bincount(classify)
		count= count.reshape(len(count),1)
		for i in range(self.X.shape[0]):
			self.n_c[classify[i],:] = self.n_c[int(classify[i]),:] + self.X[i,:]
		self.n_c = self.n_c / count

	def classify(self,X,n_clusters,epoch):
		self.X = X
		self.n_clusters = n_clusters

		if self.init == "kmeans++":
			self.C = self.initialize_centroids()
			self.set_center()
		else:
			self.C = self.X[:self.n_clusters]
			self.X = self.X[self.n_clusters:]
			self.set_center()
			self.X = np.append(self.X,self.C,axis = 0)

		for e in range(epoch):
			self.set_center()
		classify = np.zeros(self.X.shape[0])
		for i in range(self.X.shape[0]):
			temp = self.dists[:,i].ravel()
			classify[i] = np.where(temp == np.amin(temp))[0]
		classify = classify.astype(int)
		prediction = []
		for i in range(self.n_clusters):
			prediction.append([])
		for i in range(self.X.shape[0]):
			prediction[classify[i]].append(list(self.X[i]))
		for i in range(self.n_clusters):
			prediction[i] = np.array(prediction[i])
		self.prediction = prediction
		return prediction

def get_data_from_file(filename):
    datafile = open(filename)
    data = []
    for row in datafile:
       	tup = []
        for ele in row.split(','):
            tup.append(float(ele))
        data.append(np.array(tup))
    return np.array(data)

#feathing data from the file
filename = 'test_dataset.txt'
X = get_data_from_file(filename = filename)

#data classification
n_clusters = 8
epoch = 10
clf = Kmeans(init="kmeans++")
clf.classify(X,n_clusters,epoch)

#data visualization
clf.visualize()
