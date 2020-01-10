import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

	def __init__(self):
		self.colors = {"cost_function":"red","prediction":"blue",'data_point':'green'}
		self.sizes = {'cost_function':1,'prediction':1,'data_point':5}

	def hθ(self):
		return self.X.dot(self.θ).reshape(self.X.shape[0],1)

	def cost_function(self):
		return np.sum(np.square(self.hθ()-self.y))/(2*self.y.size)

	def expand_feature(self,X,feature):
		return np.hstack(tuple([np.power(X,i) for i in range(feature)]))

	def gradint_descent(self):
		Δ = []
		for j in range(self.θ.size):
			Δ.append(np.sum((self.hθ()-self.y) * self.X[:,j].reshape(self.y.shape))/self.y.size)
		Δ = np.array(Δ).reshape(self.θ.shape)
		self.θ = self.θ - self.α * Δ
		return self.θ , self.cost_function()

	def fit(self,X,y,α,feature_level,epoch):
		self.X,self.y,self.α,self.feature_level,self.epoch = X,y,α,feature_level,epoch
		self.X = self.expand_feature(self.X,self.feature_level)
		self.θ = np.random.rand(self.X[0].size).reshape(self.X[0].size,1)
		self.costs = []
		for i in range(self.epoch):
			self.θ,cost= self.gradint_descent()
			self.costs.append(cost)
		return self.θ,np.array(self.costs)

	def plot(self,θ):
		X = np.linspace(0,1,1000).reshape(1000,1)
		return X,self.expand_feature(X,θ.size).dot(θ)

	def visualize(self):
		plt.xlabel('x - axis') 
		plt.ylabel('y - axis') 
		plt.title('linear regression')
		plt.scatter(self.X[:,1],self.y,color=self.colors['data_point'],label='datapoint',s=self.sizes['data_point'])
		c_x,c_y = np.linspace(0,1,epoch) , costs
		plt.plot(c_x,c_y,color=self.colors['cost_function'],label='cost function',linewidth = self.sizes['cost_function'])
		p_x,p_y = self.plot(θ)
		plt.plot(p_x,p_y,color=self.colors['prediction'],label='prediction',linewidth = self.sizes['prediction'])
		plt.legend()
		plt.show()

#fetching data from the file
datafile = open('./test_dataset.txt')
X,y= [],[]
for tup in datafile:
	x_n,y_n = tup.split()
	X.append(float(x_n))
	y.append(float(y_n))

#data preprocessing
def normalize(X,y):
	return (X-X.min())/(X.max()-X.min()),(y-y.min())/(y.max()-y.min())

X,y = np.array(X).reshape(len(X),1) , np.array(y).reshape(len(y),1)
X,y = normalize(X,y)

#fitting the data
α = 0.1
feature_level = 10
epoch = 10000
clf = LinearRegression()
θ,costs = clf.fit(X,y,α,feature_level,epoch)


#data visualization
clf.visualize()





