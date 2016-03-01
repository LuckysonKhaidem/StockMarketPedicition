from sklearn.datasets import load_digits as load
from sklearn.cross_validation import train_test_split
from DecisionTree import *
from sklearn.metrics import accuracy_score
from RF import RandomForestClassifier
#from sklearn.ensemble import RandomForestClassifier
data = load()

X = data.data
y = data.target

print X.shape
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)

dtree = RandomForestClassifier(n_estimators = 10)

dtree.fit(Xtrain,ytrain)

y_pred = dtree.predict(Xtest)

print accuracy_score(y_pred,ytest)

