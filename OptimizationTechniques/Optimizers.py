import numpy as np

def optimizers(optimizer):
	if optimizer == "adam":
		def adam(X,y,α,epoch,cost_fn):
			compute_cost = cost_functions(cost_fn)
			compute_gradient =  compute_gradient_functions(cost_fn)
			θ = np.random.rand(X[0].size).reshape(X[0].size,1)
			costs = []
			β1,β2,ε,m,v = 0.9,0.99,0.1,0,0
			for i in range(epoch):
				Δ= compute_gradient(X,y,θ)
				m = β1 * m + (1 - β1) * Δ
				v = β2 * v + (1 - β2) * np.power(Δ, 2)
				m_hat = m / (1 - np.power(β1, i+1))
				v_hat = v / (1 - np.power(β2, i+1))
				θ = θ - α * m_hat / (np.sqrt(v_hat) + ε)
				costs.append(compute_cost(X,y,θ))
			return θ,np.array(costs)
		return adam

	elif optimizer == "adagrade":
		def adagrade(X,y,α,epoch,cost_fn):
			compute_cost = cost_functions(cost_fn)
			compute_gradient =  compute_gradient_functions(cost_fn)
			θ = np.random.rand(X[0].size).reshape(X[0].size,1)
			Δ_squared = 0
			costs = []
			for i in range(epoch):
				Δ = compute_gradient(X,y,θ)
				Δ_squared += Δ * Δ
				θ = θ - (α / np.sqrt(Δ_squared)) * Δ
				costs.append(compute_cost(X,y,θ))
			return θ,np.array(costs)
		return adagrade

	elif optimizer == "rmsprop":
		def rmsprop(X,y,α,epoch,cost_fn):
			compute_cost = cost_functions(cost_fn)
			compute_gradient =  compute_gradient_functions(cost_fn)
			θ = np.random.rand(X[0].size).reshape(X[0].size,1)
			Δ_squared = 0
			costs = []
			for i in range (epoch):
				Δ = compute_gradient(X,y,θ)
				Δ_squared = 0.9 * Δ_squared + 0.1 * Δ * Δ
				θ = θ - (α/ np.sqrt(Δ_squared)) * Δ
				costs.append(compute_cost(X,y,θ))
			return θ,np.array(costs)
		return rmsprop

	elif optimizer == "amsgrade":
		def amsgrade(X,y,α,epoch,cost_fn):
			compute_cost = cost_functions(cost_fn)
			compute_gradient =  compute_gradient_functions(cost_fn)
			θ = np.random.rand(X[0].size).reshape(X[0].size,1)
			costs = []
			β1,β2,ε,m,v = 0.9,0.99,0.1,0,0
			v_hat = 0
			for i in range(epoch):
			    Δ = compute_gradient(X,y,θ)
			    m = β1 * m + (1 - β1) * Δ
			    v = β2 * v + (1 - β2) * np.power(Δ, 2)
			    v_hat = np.maximum(v, v_hat)
			    θ = θ - α * m / (np.sqrt(v_hat) + ε)
			    costs.append(compute_cost(X,y,θ))
			return θ,np.array(costs)
		return amsgrade

	elif optimizer == "nadam":
		def nadam(X,y,α,epoch,cost_fn):
			compute_cost = cost_functions(cost_fn)
			compute_gradient =  compute_gradient_functions(cost_fn)
			θ = np.random.rand(X[0].size).reshape(X[0].size,1)
			costs = []
			β1,β2,ε,m,v = 0.9,0.99,0.1,0,0
			for i in range(epoch):
			    Δ = compute_gradient(X,y,θ)
			    m = β1 * m + (1 - β1) * Δ
			    v = β2 * v + (1 - β2) * np.power(Δ, 2)
			    m_hat = m / (1 - np.power(β1, i+1)) + (1 - β1) * Δ / (1 - np.power(β1, i+1))
			    v_hat = v / (1 - np.power(β2, i+1))
			    θ = θ - α * m_hat / (np.sqrt(v_hat) + ε)
			    costs.append(compute_cost(X,y,θ))
			return θ,np.array(costs)
		return nadam

	elif optimizer == "adamax":
		def adamax(X,y,α,epoch,cost_fn):
			compute_cost = cost_functions(cost_fn)
			compute_gradient =  compute_gradient_functions(cost_fn)
			θ = np.random.rand(X[0].size).reshape(X[0].size,1)
			costs = []
			β1,β2,ε,m,v = 0.9,0.99,0.1,0,0
			for i in range(epoch):
			    Δ = compute_gradient(X,y,θ)
			    m = β1 * m + (1 - β1) * Δ
			    m_hat = m / (1 - np.power(β1, i+1))
			    v = np.maximum(β2 * v, np.abs(Δ))
			    θ = θ - α* m_hat / v
			    costs.append(compute_cost(X,y,θ))
			return θ,np.array(costs)
		return adamax

	elif optimizer == "rprop":
		def rprop(X,y,α,epoch,cost_fn):
			compute_cost = cost_functions(cost_fn)
			compute_gradient =  compute_gradient_functions(cost_fn)
			θ = np.random.rand(X[0].size).reshape(X[0].size,1)
			Δ = []
			incFactor,decFactor,step_size_max,step_size_min = 1, 1, 1, 0.0001
			costs = []
			α = np.full(θ.size,α).reshape(θ.size,1)
			Δ.append(compute_gradient(X,y,θ))
			for t in range(epoch):
			    Δ.append(compute_gradient(X,y,θ))
			    for i in range(θ.size):
				    if Δ[t+1][i] * Δ[t][i] > 0:
				        α[i] = min(α[i] * incFactor, step_size_max)
				    elif Δ[t+1][i] * Δ[t][i] < 0:
				        α[i] = max(α[i] * decFactor, step_size_min)
			    θ = θ - α * np.sign(Δ[t+1])
			    costs.append(compute_cost(X,y,θ))
			return θ,np.array(costs)
		return rprop

def cost_functions(fn):
	if fn == "MSE":
		def MSE(X,y,θ):
			return np.sum(np.square(X.dot(θ).reshape(X.shape[0],1)-y))/(2*y.size)
		return MSE

def compute_gradient_functions(fn):
	if fn == "MSE":
		def MSE(X,y,θ):
			Δ = []
			for j in range(θ.size):
				Δ.append(np.sum((X.dot(θ).reshape(X.shape[0],1)-y) * X[:,j].reshape(y.shape))/y.size)
			return np.array(Δ).reshape(θ.shape)
		return MSE
 #mean  μ
 
