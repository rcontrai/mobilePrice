from data import *
import matplotlib.pyplot as plt

datatrain = carga_csv("data/train.csv")
datatest = carga_csv("data/test.csv")
data = np.concatenate((datatrain,datatest), axis=0)
X = data[:,:-1]
Y = data[:,-1]

#2D scatter plot
fig = scatterPlot(X,Y,(13,12))
plt.title("2D plot of 2 features")
fig.show()

#3D scatter plot
fig = scatterPlot(X,Y,(13,12,0))
plt.title("3D plot of 3 features")
fig.show()

