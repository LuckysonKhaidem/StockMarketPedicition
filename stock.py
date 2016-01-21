import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc
import numpy as np
from matplotlib import pyplot as plt

def sign(x):
	if x >= 0:
		return 1
	else:
		return 0

def getData(CSVFile):
	data = pandas.read_csv(CSVFile)
	data = data[::-1]
	x = np.c_[data['Open']]
	x = np.c_[x,data['High']]
	x = np.c_[x,data['Low']]
	x = np.c_[x,data['Close']]
	x = np.c_[x,data['Volume']]
	return pandas.stats.moments.ewma(x,span = 20)


def prepareData(X,d):
	n_sample,n_features = X.shape
	x = X[:n_sample-d]
	y0 = X[:,3][:n_sample-d]
	yd = X[:,3][d:]
	y = np.array(map(sign,yd-y0))
	return x,y

CSVFile = raw_input("Enter the csv file: ")
X,y = prepareData(getData(CSVFile),90)
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
model = RandomForestClassifier(n_estimators = 20,criterion = "entropy")
model.fit(Xtrain,ytrain)
print Xtrain.shape
print Xtest.shape
y_pred = model.predict(Xtest)
print confusion_matrix(ytest,y_pred)
print "The accuracy is ",accuracy_score(ytest,y_pred)*100,"%"
fpr,tpr,thresholds = roc_curve(ytest,y_pred)
plt.plot(fpr,tpr,"r")
plt.plot([0,1],[0,1],"r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

	
	


