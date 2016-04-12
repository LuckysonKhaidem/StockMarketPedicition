import pandas 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from TechnicalAnalysis import *
from matplotlib import pyplot as plt
from DataVisualization import *
from tabulate import *
from DataFetcher import *
from sklearn.tree import export_graphviz
import re
from collections import defaultdict

def sign(x):
	if x >= 0:
		return 1
	else:
		return 0

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
	close = RSI[:,1]
	feature_matrix = np.c_[RSI[:,0],
						   StochasticOscillator[:,0],
						   Williams[:,0],
						   MACD[:,0],
						   PROC[:,0],
						   OBV[:,0],
						   close]

	return feature_matrix

def prepareData(X,d):

	feature_matrix = getTechnicalIndicators(X,d)
	num_samples = feature_matrix.shape[0]
	y0 = feature_matrix[:,-1][:num_samples-d]
	y1 = feature_matrix[:,-1][d:]
	feature_matrix = feature_matrix[:num_samples-d]
	
	
	y = np.sign(y1 - y0)

	return np.c_[feature_matrix,y1],y

def accurac_score(ytest,y_pred):

	ytest = np.array(ytest)
	y_pred = np.array(y_pred)
	ytest = ytest.squeeze()
	y_pred = y_pred.squeeze()
	matched_index = np.where(ytest == y_pred)

	total_samples = len(ytest)
	matched_samples = len(matched_index[0])
	accuracy = float(matched_samples)/total_samples
	return accuracy * 100



def confusion_matr(ytest,y_pred):

	ytest = ytest.squeeze()
	y_pred = y_pred.squeeze()
	n = len(list(set(ytest)))
	conf_mat = np.zeros((n,n))
	for i,j in zip(ytest,y_pred):
		conf_mat[i][j] += 1
	return conf_mat

def specificity_score(ytest,y_pred):

	conf_mat = confusion_matrix(ytest,y_pred)
	specificity = float(conf_mat[0][0])/(conf_mat[0][0]+conf_mat[0][1])
	return specificity
	
def visualizeTree(model):

	feature_names = ["RSI","Stochastic Oscillator","Williams","MACD","PROC","OBV"]
	i = 0
	for tree in model.estimators_:
		outfile = "ForestGDL/Tree"+str(i)+".dot"
		export_graphviz(tree,out_file = outfile,feature_names = feature_names, rounded = True, special_characters = True,class_names = ["Fall","Rise"],node_ids = True)
		i += 1

def BuildTreeStructure(GDLFile):

		file = open(GDLFile,"r")
		gdl = file.read()
		gdl = gdl.replace("[labeldistance=2.5, labelangle=45, headlabel=\"True\"] ","")
		gdl = gdl.replace("[labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ","")
		pattern0 = r"[0-9]* -> [0-9]* ;"
		directed_edges = re.findall(pattern0,gdl)
		children = defaultdict(int)
		split_decision = defaultdict(int)
		class_label = defaultdict(int)
		for edge in directed_edges:

			edge = edge.replace(" ;","")
			parent,child = map(int,edge.split(" -> "))

			if parent not in children.keys():
				children[parent] = []
			children[parent].append(child)

		pattern1 = r"[0-9]+ \[.*\] ;"
		node_description = re.findall(pattern1,gdl)

		for desc in node_description:
			pattern = re.compile("^([0-9]*) (\[.*\]) ;")
			node_id = int(pattern.search(desc).group(1))
			label = pattern.search(desc).group(2)
			pattern2 = re.compile("<br/>(.*) &le; (-?[0-9]+\.[0-9]+)<br/>")
			
			result = pattern2.search(label)
			if result:
				feature, threshold = result.group(1),float(result.group(2))
				split_decision[node_id] = (feature,threshold)
			pattern3 = re.compile("<br/>class = (.*)>]")
			result = pattern3.search(label)
			if result:
				class_label[node_id] = result.group(1)
		
		return children,split_decision,class_label	

def trace(X,y):


	X = list(X)
	print X
	label = {-1:"Fall",1:"Rise"}
	f_index = {"RSI":0,
			"Stochastic Oscillator":1,
			"Williams":2,
			"MACD":3,
			"PROC":4,
			"OBV":5}
	vote_count = [0,0]
	for i in range(30):

		GDLFile = "ForestGDL/Tree"+str(i)+".dot"
		children,split_decision,class_label = BuildTreeStructure(GDLFile)
		print "For Tree "+str(i)+":\\\\"
		cur_node = 0

		while children[cur_node] != 0:

			feature = split_decision[cur_node][0]
			threshold = split_decision[cur_node][1]
			index = f_index[feature]
			print "At node "+str(cur_node)+":("+feature+"="+str(X[index])+") \\textless= "+str(threshold)+"?\\\\"
			if X[index] <= threshold:
				cur_node = min(children[cur_node])
				print "True: Go to Node",cur_node,"\\\\"
			else:
				cur_node = max(children[cur_node])
				print "False: Go to Node",cur_node,"\\\\"

		print "Leaf Node",cur_node,"is labeled as",class_label[cur_node],"\\\\"
		print "\\\\"
		print "\\\\"
		if class_label[cur_node] == "Fall":
			vote_count[0] += 1
		else:
			vote_count[1] += 1

	print "Vote Count",vote_count
	print "Actual output: ",label[y]

def main(CSVFile):

	
	table = []
	ohclv_data = getData(CSVFile)
	for Trading_Day in [30]:
		X,y = prepareData(ohclv_data, Trading_Day)
		Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
		model = RandomForestClassifier(n_estimators = 30,criterion = "gini")
		model.fit(Xtrain[:,range(6)],ytrain)
		
		print (Xtrain.shape[0] + Xtest.shape[0] == X.shape[0])
		y_pred = model.predict(Xtest[:,range(6)])
		accuracy = accuracy_score(ytest,y_pred)
		precision = precision_score(ytest,y_pred)
		recall = recall_score(ytest,y_pred)
		specificity = specificity_score(ytest,y_pred)
		day = str(Trading_Day)+" days"
		row = [day,accuracy,precision,recall,specificity]
		table.append(row)

	#headers = ["Time Window","Accuracy","Precision","Recall","Specificity"]
	#print tabulate(table,headers,tablefmt="fancy_grid")
	visualizeTree(model)
	trace(Xtest[0],ytest[0])
		
def TimePeriodvsAccuracy(CSVFile):

	ochlv_data = getData(CSVFile)
	timeWindow = range(1,91)
	accuracies = []
	for day in timeWindow:
		X,y = prepareData(ochlv_data, day)
		Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
		model = RandomForestClassifier(n_estimators = 30, criterion = "gini")
		model.fit(Xtrain[:,range(6)],ytrain)
		y_pred = model.predict(Xtest[:,range(6)])
		accuracy = accuracy_score(ytest,y_pred)	
		accuracies.append(accuracy)
		print accuracy
	plt.plot(timeWindow,accuracies,linewidth=2)
	plt.xlabel("Time Window (Days)")
	plt.ylabel("Accuracy")
	plt.savefig("tva.png")
	plt.close()

def newTest(CSVFile):

	ochlv_data = getData(CSVFile)
	Trading_Day = 1
	model = RandomForestClassifier(n_estimators = 30, criterion = "gini")
	X,y = prepareData(ochlv_data,Trading_Day)
	Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
	print (Xtrain.shape[0] + Xtest.shape[0] == X.shape[0])
	model.fit(Xtrain[:,range(6)],ytrain)
	y_pred = model.predict(Xtest[:,range(6)])
	accuracy = accuracy_score(ytest,y_pred)*100
	index = np.arange(2)
	fig,ax = plt.subplots()
	barwidth = 0.35
	
	accuracies = [81.56,accuracy]  # bring some raw data

	accu_series = pandas.Series.from_array(accuracies)   # in my original code I create a series and run on that, so for consistency I create a series from the list.

	x_labels = ["CS-SVM","RF"]

	# now to plot the figure...
	plt.figure(figsize=(12, 8))
	ax = accu_series.plot(kind='bar')
	ax.set_title("Accuracy comparision")
	ax.set_xlabel("Algorithm")
	ax.set_ylabel("Accuracy %")
	ax.set_xticklabels(x_labels)

	rects = ax.patches

	# Now make some labels
	labels = [str(i) for i in accuracies]

	for rect, label in zip(rects, labels):
	    height = rect.get_height()
	    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
	plt.savefig("nifty.png")



Fetcher = DataFetcher()
stock_symbol = "AAPL"
#stock_symbol = raw_input("Enter the stock_symbol: ")
#Fetcher.getHistoricalData(stock_symbol)
CSVFile = stock_symbol+".csv"
main(CSVFile)

	
	


