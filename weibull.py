import numpy as np
import pandas 
from matplotlib import pyplot as plt

def weib(x,n,a):
	return (a/n) *((x/n)**(a-1)) * np.exp(-(x/n)**a)


data = pandas.read_csv("apple.csv")
data = data[::-1]
volume = np.array(data["Volume"])
fx = weib(np.array(volume),2.0,1.0)
print fx
plt.plot(fx)
plt.show()
