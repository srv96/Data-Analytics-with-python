import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import Optimizers

class LinearRegression:

	def __init__(self):
		self.colors = {"cost_function":"red","prediction":"blue",'data_point':'green'}
		self.sizes = {'cost_function':1,'prediction':1,'data_point':5}

	def expand_feature(self,X,feature):
		return np.hstack(tuple([np.power(X,i) for i in range(feature)]))

	def fit(self,X,y,α,epoch,optimizer,cost_function,etsimation,feature_level):
		self.X,self.y,self.α,self.feature_level,self.epoch = X,y,α,feature_level,epoch
		if etsimation == "quadratic":
			self.X = self.expand_feature(self.X,self.feature_level)
		self.optimizer = Optimizers.optimizers(optimizer)
		self.θ,self.costs = self.optimizer(self.X,self.y,α,epoch,cost_function)
		return self.θ,self.costs

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
feature_level = 3
epoch = 1000
optimizer ="adam"
estimation = "quadratic"
cost_function = "MSE"
clf = LinearRegression()
θ,costs = clf.fit(X,y,α,epoch,optimizer,cost_function,estimation,feature_level)
print(costs[epoch-1])

#data visualization
clf.visualize()





