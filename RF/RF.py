from DecisionTree import *
import numpy as np 
from scipy.stats import mode
#from sklearn.tree import DecisionTreeClassifier as Tree

class RandomForestClassifier:
	def __init__(self,n_estimators = 10, criterion = "gini"):

		self.n_estimators = n_estimators
		self.criterion = criterion
		self.forest = []

	def fit(self,X,y):

		n_sample,n_feature = X.shape
		

		#low = int(0.3 * n_sample)
		#high = n_sample
		subspace_size = int(0.9 * n_sample)

		self.tree_info = []

		for i in xrange(self.n_estimators):

			sample_index = np.random.choice(range(n_sample),subspace_size)
			feature_size = np.random.randint(1,n_feature+1,1)
			feature_index = np.random.choice(range(n_feature),feature_size)
			self.tree_info.append((sample_index,feature_index))
			#X_1 = X[sample_index]
			#y_1 = y[sample_index]
			
			#X_1 = X_1[:,feature_index]
			tree = Tree(sample_index = sample_index, feature_index =feature_index)
			tree.fit(X,y)
			self.forest.append(tree)
			

	def predict(self,X):

		trees_output = []
		for tree in self.forest:
			
			#sample_index = info[0]
			#feature_index = info[1]
			#X_1 = X[:,feature_index]
			row_output = tree.predict(X)
			trees_output.append(row_output)
		trees_output = np.array(trees_output)
		output = mode(trees_output,axis = 0)
		

		return output[0][0]







