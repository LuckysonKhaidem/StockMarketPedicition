import numpy as np
import pandas
from FeatureExtraction import *

data = pandas.read_csv("rs.csv")
x = np.c_[range(1,31)]
x = np.c_[x,data["High"]]
x = np.c_[x,data[" Low"]]
x = np.c_[x,data["Current Close"]]
print getPriceRateo
