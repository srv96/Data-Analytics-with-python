import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optimizers
import losses
import models
import preprocess


#fetching data from the file
df = pd.read_csv('./datasets/test_data.csv')
X,y = np.array(df['X']),np.array(df['y'])

#data preprocessing
def normalize(X,y):
	return (X-X.min())/(X.max()-X.min()),(y-y.min())/(y.max()-y.min())

X,y = np.array(X).reshape(len(X),1) , np.array(y).reshape(len(y),1)
X,y = normalize(X,y)


#fitting the data
α = 0.1
feature_level = 5
epoch = 100
estimation = "quadratic"
optimizer = 'sgd'
loss = 'MSE'
reg = models.Regressor()
reg.compile(opt = optimizer,loss=loss)
costs = reg.fit(X,y,α,epoch,estimation,feature_level)
θ = reg.get_params()


#data visualization
plt.plot(np.arange(epoch),costs)
plt.show()