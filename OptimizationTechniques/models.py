import numpy as np
import matplotlib.pyplot as plt
import optimizers
import losses

class Regressor:

	def __init__(self):
		self.colors = {"cost_function":"red","prediction":"blue",'data_point':'green'}
		self.sizes = {'cost_function':1,'prediction':1,'data_point':5}
		self.costs = []
		self.loss = None
		self.optimizer = None
		self.θ = None
		self.Δ = None

	def expand_feature(self,X,feature):
		return np.hstack(tuple([np.power(X,i) for i in range(feature)]))

	def compile(self,opt,loss):
		if type(opt) == str :
			self.optimizer = optimizers.get_default_instance(opt)
		else:
			self.optimizer = opt
		if type(loss) == str:
			self.loss = losses.get_default_instance(loss)
		else:
			self.loss = loss

	def fit(self,X,y,α,epoch,etsimation,feature_level):
		self.X,self.y,self.α,self.feature_level,self.epoch = X,y,α,feature_level,epoch

		if etsimation == "quadratic":
			self.X = self.expand_feature(self.X,self.feature_level)

		self.θ = np.random.rand(self.X[0].size).reshape(self.X[0].size,1)

		for i in range(epoch):
			self.Δ = self.loss.grad(self.X,self.y,self.θ)
			self.θ = self.optimizer.optimize(self.X,self.y,self.θ,self.Δ)
			self.costs.append(self.loss.loss(self.X,self.y,self.θ))

		return np.array(self.costs)

	def get_hyperplane(self,θ):
		X = np.linspace(0,1,1000).reshape(1000,1)
		return X,self.expand_feature(X,θ.size).dot(θ)

	def predict(self,X):
		return self.expand_feature(X,θ.size).dot(θ)

	def get_params(self):
		return self.θ






