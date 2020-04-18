import numpy as np

def get_default_instance(loss):
	if loss == 'MSE':
		return MSE()


class Loss : 
	def loss(X,y,θ) -> np.ndarray:
		pass
	def grad(X,y,θ) ->np.ndarray:
		pass

class MSE(Loss) :
	def loss(self,X,y,θ):
		return np.sum(np.square(X.dot(θ).reshape(X.shape[0],1)-y))/(2*y.size)

	def grad(self,X,y,θ):
		Δ = []
		for j in range(θ.size):
			Δ.append(np.sum((X.dot(θ).reshape(X.shape[0],1)-y) * X[:,j].reshape(y.shape))/y.size)
		return np.array(Δ).reshape(θ.shape)