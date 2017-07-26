from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.metrics import roc_curve,auc
import os
from sklearn.tree import export_graphviz
from subprocess import check_call

def SegragateData(X,y):
	
	n = len(y)
	X0 = []
	X1 = []

	for i in xrange(n):
		if y[i] == 0:
			X0.append(X[i])

		else:
			X1.append(X[i])

	return np.array(X0),np.array(X1)

def ScatterPlot(X,y):
	c = np.array(["b","y"])
	pca = PCA(n_components = 2)
	pca.fit(X)
	pca = pca.transform(X)
	X0,X1 = SegragateData(X,y)
	np.savetxt("class1.csv",X0,delimiter = ",")
	np.savetxt("class2.csv",X1,delimiter = ",")
	
def DrawConvexHull(X,y):

	ScatterPlot(X,y)
	pca = PCA(n_components = 2)
	pca.fit(X)
	X = pca.transform(X)
	X0,X1 = SegragateData(X,y)
	hull0 = ConvexHull(X0)
	hull1 = ConvexHull(X1)
	X_0 = list(X0[hull0.vertices,0])
	X_0.append(X_0[0])
	Y_0 = list(X0[hull0.vertices,1])
	Y_0.append(Y_0[0])
	X_1 = list(X1[hull1.vertices,0])
	X_1.append(X_1[0])
	Y_1 = list(X1[hull1.vertices,1])
	Y_1.append(Y_1[0])
	C0 = np.c_[X_0,Y_0]
	C1 = np.c_[X_1,Y_1]
	np.savetxt("hull1.csv",C0,delimiter = ",")
	np.savetxt("hull2.csv",C1,delimiter = ",")
	





	
