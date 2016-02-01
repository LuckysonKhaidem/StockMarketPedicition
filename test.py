from ANN import *
from sklearn.datasets import load_digits as load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

data = load_iris()
X = data.data
Y = data.target
Xtrain, Xtest, Ytrain,Ytest = train_test_split(X,Y)
model = NeuralNetwork(Xtrain,Ytrain)
model.fit(0,0)
y_pred = model.predict(Xtest)
print accuracy_score(y_pred,Ytest)


