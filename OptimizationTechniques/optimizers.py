import sys
import inspect
import numpy as np

def get_default_instance(opt):
	clsmem = inspect.getmembers(sys.modules[__name__],inspect.isclass)
	all_class = {key.lower():value for key,value in clsmem}
	return all_class[opt]()

class Optimizer :
	def optimize(self,X,y,epoch) -> np.ndarray :
		pass
	def get_config(self) -> dict :
		pass

class AdaDelta(Optimizer):
	def __init__(self, μ = 0.95, ε = 1e-05,α = 1,decay = 0):
		self.μ = μ
		self.ε = ε
		self.α = α
		self.decay = decay
		self.EΔ2, self.Δx, self.EΔx2, self.iterations = 0, 0, 0, 0
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		self.EΔ2 = self.μ * self.EΔ2 + (1 - self.μ) * np.square(Δ)
		self.Δx = np.sqrt(self.EΔx2 + self.ε) * (Δ / np.sqrt(self.EΔ2 + self.ε)) 
		self.EΔx2 = self.μ * self.EΔx2 + (1 - self.μ) * np.square(self.Δx)
		θ = θ - self.α * self.Δx
		self.α = self.α * (1 / (1 + self.decay * self.iterations))
		self.iterations = self.iterations + 1
		return θ		

class Adagrade(Optimizer):

	def __init__(self,α=0.36):
		self.α = α
		self.Δ_squared = 0
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		self.Δ_squared += np.square(Δ)
		θ = θ - (self.α / np.sqrt(self.Δ_squared)) * Δ
		return θ

	def get_config(self):
		return self.config


class Adam(Optimizer):

	def __init__(self,α=0.36,decay =1e-05,β1=0.9,β2=0.99,ε=1e-08):
		self.α = α
		self.decay = decay
		self.β1 = β1
		self.β2 =β2
		self.ε = ε
		self.m , self.v = 0 , 0 
		self.iterations = 0
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		self.m = self.β1 * self.m + (1 - self.β1) * Δ
		self.v = self.β2 * self.v + (1 - self.β2) * np.power(Δ, 2)
		self.m_hat = self.m / (1 - np.power(self.β1,self.iterations+1))
		self.v_hat = self.v / (1 - np.power(self.β2,self.iterations+1))
		self.iterations = self.iterations + 1
		θ = θ - self.α * self.m_hat / (np.sqrt(self.v_hat) + self.ε)
		return θ

	def get_config(self):
		return self.config

class AdaMax(Optimizer):

	def __init__(self,α=0.36,β1=0.9,β2=0.99,ε=1e-04,decay=1e-05):
		self.α = α
		self.β1 = β1
		self.β2 = β2
		self.ε = ε
		self.m ,self.v,self.v_hat ,self.iterations = 0 , 0 , 0 , 0
		self.decay = decay
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		self.m = self.β1 * self.m + (1 - self.β1) * Δ
		self.m_hat = self.m / (1 - np.power(self.β1, self.iterations+1))
		self.v = np.maximum(self.β2 * self.v, np.abs(Δ))
		θ = θ - self.α* self.m_hat / self.v
		self.α = self.α* (1. / (1. + self.decay * self.iterations))
		self.iterations = self.iterations + 1
		return θ

	def get_config(self):
		return self.config

class AMSgrade(Optimizer):

	def __init__(self,α=0.36,β1=0.9,β2=0.99,ε=1e-04,decay=1e-05):
		self.α = α
		self.β1 = β1
		self.β2 = β2
		self.ε = ε
		self.m ,self.v,self.v_hat ,self.iterations = 0 , 0 , 0 , 0
		self.decay = decay
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		self.m = self.β1 * self.m + (1 - self.β1) * Δ
		self.v = self.β2 * self.v + (1 - self.β2) * np.power(Δ, 2)
		self.v_hat = np.maximum(self.v, self.v_hat)
		θ = θ - self.α * self.m / (np.sqrt(self.v_hat) + self.ε)
		self.α = self.α* (1. / (1. + self.decay * self.iterations))
		self.iterations = self.iterations + 1
		return θ

	def get_config(self):
		return self.config

class NAdam(Optimizer):

	def __init__(self,α=0.36,β1=0.9,β2=0.99,ε=1e-04,decay=1e-05):
		self.α = α
		self.β1 = β1
		self.β2 = β2
		self.ε = ε
		self.m ,self.v,self.v_hat ,self.iterations = 0 , 0 , 0 , 0
		self.decay = decay
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		self.m = self.β1 * self.m + (1 - self.β1) * Δ
		self.v = self.β2 * self.v + (1 - self.β2) * np.power(Δ, 2)
		self.m_hat = self.m / (1 - np.power(self.β1, self.iterations+1)) + (1 - self.β1) * Δ / (1 - np.power(self.β1, self.iterations+1))
		self.v_hat = self.v / (1 - np.power(self.β2, self.iterations+1))
		θ = θ - self.α * self.m_hat / (np.sqrt(self.v_hat) + self.ε)
		self.α = self.α* (1. / (1. + self.decay * self.iterations))
		self.iterations = self.iterations + 1
		return θ

	def get_config(self):
		return self.config


class RMSprop(Optimizer):

	def __init__(self,α=0.01,μ=0.9,decay=1e-04):
		self.decay = decay
		self.α = α
		self.Δ_squared = 0
		self.μ = μ
		self.iterations = 0
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		self.α = self.α* (1. / (1. + self.decay * self.iterations))
		self.Δ_squared = self.μ* self.Δ_squared + (1- self.μ) * np.square(Δ)
		θ = θ - (self.α / np.sqrt(self.Δ_squared)) * Δ
		self.iterations = self.iterations + 1
		return θ

	def get_config(self):
		return self.config


class Rprop(Optimizer):

	def __init__(self,α=0.36,incFactor=1,decFactor=1,step_size_max=1,step_size_min=1e-05,decay = 1e-03):
		self.Δ = []
		self.incFactor = incFactor
		self.decFactor = decFactor
		self.step_size_max = step_size_max
		self.step_size_min = step_size_min
		self.α = α
		self.iterations = 0
		self.decay = decay
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		if self.iterations == 0 :
			self.Δ.append(Δ)
			self.α = np.full(θ.size,self.α).reshape(θ.size,1)
		self.Δ.append(Δ)
		for i in range(θ.size):
			if self.Δ[self.iterations+1][i] * self.Δ[self.iterations][i] > 0:
				self.α[i] = min(self.α[i] * self.incFactor, self.step_size_max)
			elif self.Δ[self.iterations+1][i] * self.Δ[self.iterations][i] < 0:
				self.α[i] = max(self.α[i] * self.decFactor, self.step_size_min)
		θ = θ - self.α * np.sign(self.Δ[self.iterations+1])
		self.α = self.α* (1. / (1. + self.decay * self.iterations))
		self.iterations = self.iterations + 1
		return θ

	def get_config(self):
		return self.config


class SGD(Optimizer):
	def __init__(self,μ = 0.9,α = 0.36,nesterov =True,decay = 1e-04):
		self.μ = μ
		self.α = α
		self.iterations = 0
		self.nesterov = nesterov
		self.decay = decay
		self.config = locals()
		del self.config['self']

	def optimize(self,X,y,θ,Δ):
		self.α = self.α* (1. / (1. + self.decay * self.iterations))
		if self.nesterov :
			if self.iterations == 0 :
				self.θ_nxt = θ - self.α * Δ
				self.iterations = self.iterations + 1
				self.y_nxt = self.θ_nxt + self.μ * (self.θ_nxt - θ)
				return self.y_nxt
			else :
				self.θ_nxt = self.y_nxt - self.α * Δ
				self.y_nxt = self.θ_nxt + self.μ * (self.θ_nxt - θ)
				return self.y_nxt
		else:
			self.Δ = np.zeros(θ.shape)
			self.Δ = self.μ * self.Δ + (1- self.μ) * Δ
			θ = θ - self.α * self.Δ
		return θ

	def get_config(self):
		return self.config

