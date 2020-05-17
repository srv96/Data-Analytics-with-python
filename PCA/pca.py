import numpy as np
import pandas as pd

class PCA:
	def __init__(self,n_component):
		self.n_component = n_component

	def __covarience_matrix(self,X):
		return X.T.dot(X)

	def __p_component_values(self,COV):
		eigenValues, eigenVectors = np.linalg.eig(COV)
		idx = eigenValues.argsort()[::-1]
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:, idx]
		return eigenValues, eigenVectors

	def fit(self,X):
		self.COV = self.__covarience_matrix(X)
		self.eigenValues,self.eigenVectors = self.__p_component_values(self.COV)

	def transform(self,X):

		self.eigen_pairs = np.concatenate((np.abs(self.eigenValues).reshape(self.eigenValues.size,1),self.eigenVectors), axis=1)
		self.eigen_pairs = np.array(sorted(self.eigen_pairs, key=lambda a_entry: a_entry[0],reverse = True))
		self.eigenVectors = self.eigen_pairs[:,1:]

		if n_component > 0:
			eigenVectors = np.delete(self.eigenVectors, range(self.n_component,self.eigenVectors.shape[1]), axis=0)[::-1]
			X_transform = X.dot(eigenVectors.T)
			
		else:
			raise Exception("Minimum dimension should be more than 0")

		return X_transform

	def fit_transform(self,X):
		self.fit(X)
		return self.transform(X)



def StandardScalar(X, centering=True, scaling=True):
    X = X.astype(float)

    if centering:
        X = X - X.mean(axis = 0)
    if scaling:
        X = X / np.std(X, axis=0)

    return X

#data preprocess
df = pd.read_csv('./datasets/test_data.csv')
X = np.array(df)
X = StandardScalar(X)

#pricipal component analysis
n_component = 2
pca = PCA(n_component=2)
X_transform = pca.fit_transform(X)


