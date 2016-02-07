from ANN import *
from sklearn.datasets import  load_digits as load
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from nolearn.dbn import DBN


data = load()
X = data.data
Y = data.target
Xtrain, Xtest, Ytrain,Ytest = train_test_split(X,Y)
model = DBN([Xtrain.shape[1],100,10],
	learn_rates = 0.01,
	learn_rate_decays = 0.9,
	epochs = 20,
	verbose = 1
)
model.fit(Xtrain,Ytrain)
y_pred = model.predict(Xtest)
print accuracy_score(y_pred,Ytest)


