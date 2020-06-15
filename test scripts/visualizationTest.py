from data import *

data = carga_csv("data/train.csv")
X = data[:,:-1]
Y = data[:,-1]

fig = scatterPlot(X,Y,(12,2))
fig.show()