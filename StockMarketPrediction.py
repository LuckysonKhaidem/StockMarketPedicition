import pandas
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve,auc
from TechnicalAnalysis import *
from matplotlib import pyplot as plt
from DataVisualization import *
from RF import RF
def sign(x):
	if x >= 0:
		return 0
	else:
		return 1

def ES(x,alpha):
	
	ft = x[0]
	f = [ft]
	n = x.shape[0]
	for i in xrange(1,n):
		ft_1 = alpha * x[i] + (1 - alpha) * ft
		f.append(ft_1)
		ft = ft_1
	f = np.array(f)
	return f
	
def getData(CSVFile):

	data = pandas.read_csv(CSVFile)
	data = data[::-1]
	ohclv_data = np.c_[data['Open'],
					   data['High'],
					   data['Low'],
					   data['Close'],
					   data['Volume']]
	smoothened_ohclv_data = pandas.stats.moments.ewma(ohclv_data,span = 20)
	#smoothened_ohclv_data = ES(ohclv_data,0.15)
	return  smoothened_ohclv_data

def getTechnicalIndicators(X,d):

	RSI = getRSI(X[:,3])
	StochasticOscillator = getStochasticOscillator(X)
	Williams = getWilliams(X)
	MACD = getMACD(X[:,3])
	PROC = getPriceRateOfChange(X[:,3],d)
	OBV = getOnBalanceVolume(X)

	min_len = min(len(RSI),
				  len(StochasticOscillator),
				  len(Williams),
				  len(MACD),
				  len(PROC),
				  len(OBV))

	RSI = RSI[len(RSI) - min_len:]
	StochasticOscillator = StochasticOscillator[len(StochasticOscillator) - min_len:]
	Williams = Williams[len(Williams) - min_len: ]
	MACD = MACD[len(MACD) - min_len:]
	PROC = PROC[len(PROC) - min_len:]
	OBV = OBV[len(OBV) - min_len:]

	feature_matrix = np.c_[RSI,
						   StochasticOscillator,
						   Williams,
						   MACD,
						   PROC,
						   OBV]

	return feature_matrix

def prepareData(X,d):

	feature_matrix = getTechnicalIndicators(X,d)
	n_sample, n_features = X.shape
	num_samples, n_features = feature_matrix.shape
	X = X[n_sample - num_samples:]
	feature_matrix = feature_matrix[:num_samples -d]
	y0 = X[:,3][:num_samples - d]
	y1 = X[:,3][d:]
	y = np.array(map(sign, y1 - y0))
	#y = np.sign(y1 - y0)
	return feature_matrix,y

def accuracy_score(ytest,y_pred):

	ytest = ytest.squeeze()
	y_pred = y_pred.squeeze()
	matched_index = np.where(ytest == y_pred)

	total_samples = len(ytest)
	matched_samples = len(matched_index[0])
	accuracy = float(matched_samples)/total_samples
	return accuracy * 100

def confusion_matrix(ytest,y_pred):

	ytest = ytest.squeeze()
	y_pred = y_pred.squeeze()
	n = len(list(set(ytest)))
	conf_mat = np.zeros((n,n))
	for i,j in zip(ytest,y_pred):
		conf_mat[i][j] += 1
	return conf_mat

def main():

	CSVFile = raw_input("Enter the CSV File: ")
	Trading_Day = input("Enter the Trading Day: ")
	ohclv_data = getData(CSVFile)
	X,y = prepareData(ohclv_data, Trading_Day)
	print X.shape
	Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
	model = RF.RandomForestClassifier(n_estimators = 30,criterion = "gini")
	model.fit(Xtrain,ytrain)

	y_pred = model.predict(Xtest)
	print "The accuracy is",accuracy_score(ytest,y_pred),"%"
	print confusion_matrix(ytest,y_pred)
	#DrawConvexHull(Xtest,ytest)

main()

	
	


