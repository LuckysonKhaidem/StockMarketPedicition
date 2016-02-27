from itertools import combinations
import numpy as np
from tabulate import *
from StockMarketPrediction import getData,prepareData
from scipy.stats import shapiro
import pandas
from scipy.stats import *
features = [
		"Relative Strength Index",
		"Stochastic Oscillator",
		"Williams",
		"Moving Average Convergence Divergence",
		"Price Rate of Change",
		"On Balance Volume"
	   ]

def ComputeCorrelationCoefficient(x,y):

	x_mean = x.mean()
	y_mean = y.mean()
	x_std = x.std()
	y_std = y.std()
	cov = ((x - x_mean) * (y - y_mean)).mean()
	cor = cov/(x_std * y_std)
	
	return cor

def CrossCorrelation(X):

	n = X.shape[1]
	comb = combinations(range(n),2)
	table = []

	for i,j in comb:
		current_cor = ComputeCorrelationCoefficient(X[:,i],X[:,j])
		row = [features[i],features[j],current_cor]
		table.append(row)

	headers = ["Feautre 1","Feature 2","Correlation Coefficient"]
	print "Cross Correlation Coefficient Table"
	print tabulate(table,headers,tablefmt = "fancy_grid")


def anderson(x):

	y = x.squeeze()
	y.sort()
	N = len(y)
	xbar = x.mean()
	s = x.std()
	w = (y-xbar)/s
	z = norm.cdf(w)
	i = np.arange(1,N+1)

	S = ((2*i-1.0)/N*(np.log(z)+np.log(1-z[::-1]))).sum()
	A2 = -N-S
				
	return A2
	
def ShapiroWilkTest(X):

	n = X.shape[1]
	table = []
	for i in xrange(n):
		try:
			results = shapiro(X[:,i][:4000])	
		except:
			pass
		row = [features[i],results[0],results[1]]
		table.append(row)
	headers = ["Features","W","p-value"]
	print "Shapiro-wilk test"
	print tabulate(table,headers,tablefmt = "fancy_grid")

def AndersonDarlingTest(X):

	print "Andersion Darling Test"
	n = X.shape[1]
	critical_values = [ 0.575, 0.655,  0.786,  0.917,  1.091]
	significance_levels = [ 15.,   10.,    5.,    2.5,   1. ]
	table1 =[]
	table2 = []
	for x,y in zip(critical_values,significance_levels):
		table1.append([x,y])

	print "if A^2 is greater than the critical value for the correspoinding significance level, reject H0"
	print "Critical Value vs Significance level"
	headers1 = ["Critical Value","Significance Level"]
	print tabulate(table1,headers1,tablefmt = "fancy_grid")

	for i in xrange(n):
		results = anderson(X[:,i])	
		row = [features[i],results]
		table2.append(row)

	headers2 = ["Feautres","A^2"]
	print tabulate(table2,headers2,tablefmt = "fancy_grid")
	
X = pandas.read_csv("Features.csv")
X = np.array(X)

CrossCorrelation(X)
print "\n"
ShapiroWilkTest(X)
print "\n"		
AndersonDarlingTest(X)	
