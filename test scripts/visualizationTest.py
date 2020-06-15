from data import *

data = carga_csv("data/train.csv")
X = data[:,:-1]
Y = data[:,-1]

#2D scatter plot
fig = scatterPlot(X,Y,(13,12))
fig.show()

#3D scatter plot
fig = scatterPlot(X,Y,(13,12,0))
fig.show()