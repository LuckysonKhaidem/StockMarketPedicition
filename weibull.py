import scipy.stats as s
import numpy as np
from matplotlib import pyplot as plt 
import pandas

def weib(x,n,a):
	return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


data = pandas.read_csv("apple.csv")
data = np.log(np.array(data["Volume"]))
(loc,scale) = s.exponweib.fit_loc_scale(data,1,1)
print loc,scale
x = np.linspace(data.min(),data.max())
print weib(x,loc,scale)
plt.plot(x,weib(x,scale,loc))
plt.show()

