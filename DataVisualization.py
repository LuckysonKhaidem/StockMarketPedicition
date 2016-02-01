from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import numpy as np

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

	pca = PCA(n_components = 2)
	pca.fit(X)
	pca = pca.transform(X)
	plt.scatter(pca[:,0],pca[:,1],c = y)
	plt.show()

def DrawConvexHull(X,y):

	pca = PCA(n_components = 2)
	pca.fit(X)
	X = pca.transform(X)
	X0,X1 = SegragateData(X,y)
	hull0 = ConvexHull(X0)
	hull1 = ConvexHull(X1)
	plt.plot(X0[hull0.vertices,0],X0[hull0.vertices,1],"r--")
	plt.plot(X1[hull1.vertices,0],X1[hull1.vertices,1],"b--")
	plt.show()

def DrawROC(Ytest,Y_pred):
	fpr,tpr,thresholds = roc_curve(ytest,y_pred)
	plt.plot(fpr,tpr,"r")
	plt.plot([0,1],[0,1],"r--")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.show()
		
