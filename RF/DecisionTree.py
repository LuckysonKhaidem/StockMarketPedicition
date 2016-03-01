import numpy as np
from collections import defaultdict
from itertools import combinations
from DecisionNode import DecisionNode
from SplittingThread import *

class Tree:

	def __init__(self,sample_index,feature_index):
		
		self.root = None
		self.sample_index = sample_index
		self.feature_index = feature_index

	def BinarySplit(self, Dataset, attr_index, value):

		true_index = np.where(Dataset[:,attr_index] >= value)
		false_index = np.where(Dataset[:,attr_index] < value)

		set1 = Dataset[true_index]
		set2 = Dataset[false_index]

		return set1,set2

	def CountClassPopulation(self,Dataset):

		class_labels = Dataset[:,-1]
		result = defaultdict(int)
		for label in class_labels:
			result[label] += 1

		return result

	def GiniImpurity(self,Dataset):

		gini = 0.0
		class_count = self.CountClassPopulation(Dataset)
		total_population = len(Dataset)

		class_labels = class_count.keys()
		comb = combinations(class_labels,2)

		for i,j in comb:

			p1 = float(class_count[i])/total_population
			p2 = float(class_count[j])/total_population

			gini = gini + p1*p2

		return gini

	def Entropy(self,Dataset):

		ent = 0.0
		class_count = self.CountClassPopulation(Dataset)
		total_population = len(Dataset)

		for count in class_count.values():

			p = float(count) / total_population
			ent = ent - np.log(count)/np.log(2)

		return ent

	def BuildTree(self, Dataset, func = "gini"):

		
		if len(Dataset) == 0 : return DecisionNode()

		if func == "gini":
			score_func = self.GiniImpurity
		else:
			score_func = Entropy

		current_score = score_func(Dataset)

		best_gain = 0.0
		best_criteria = None
		best_sets = None

		n_sample,n_features = Dataset.shape

		n_features -= 1

		for attr_index in xrange(0,n_features):

			value_list = np.unique(Dataset[:,attr_index])

			for value in value_list:

				set1,set2 = self.BinarySplit(Dataset,attr_index,value)
				p = float(len(set1))/n_sample

				gain = current_score - p*score_func(set1) - (1-p)*score_func(set2) 

				if gain > best_gain and len(set1) > 0 and len(set2) > 0:

					best_gain = gain
					best_criteria = (attr_index,value)
					best_sets = (set1,set2)

		if best_gain > 0.0:

			Splitting_Thread1 = SplittingThread(self.BuildTree,best_sets[0])
			Splitting_Thread2 = SplittingThread(self.BuildTree,best_sets[1])
			col = best_criteria[0]
			value = best_criteria[1]
			Splitting_Thread1.start()
			Splitting_Thread2.start()
			true_branch = Splitting_Thread1.join()
			false_branch = Splitting_Thread2.join()

			return DecisionNode(col = col,
								value = value,
								tb = true_branch,
								fb = false_branch)
		else:
			return DecisionNode(results = self.CountClassPopulation(Dataset), isLeaf = True)


	def fit(self,X,y):

		X = X[self.sample_index]
		y = y[self.sample_index]
		X = X[:,self.feature_index]
		Dataset = np.c_[X,y]
		print "Building Trees....."
		self.root = self.BuildTree(Dataset)
		return self

	def classify(self,root,row):
		
		if root.isLeaf == False:
			attr_index = root.col
			attr_value = root.value
			if row[attr_index] >= attr_value:
				output_label = self.classify(root.true_branch, row)
			else:
				output_label = self.classify(root.false_branch, row)
		else:
			results = root.results
			max_pop = 0.0
		
			for key in results.keys():
				if results[key] > max_pop:
					max_pop = results[key]
					output_label = key
		
		return output_label



	def predict(self,X):

			
			X = X[:,self.feature_index]
			if self.root == None:
				print "The model has not been trained yet"
				return
			
			y_pred = []

			for row in X:
				predicted_label = self.classify(self.root,row)
				y_pred.append(predicted_label)

			return np.array(y_pred)








		



