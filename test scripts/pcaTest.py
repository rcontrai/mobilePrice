from data import *
import matplotlib.pyplot as plt

datatrain = carga_csv("data/train.csv")
datatest = carga_csv("data/test.csv")
data = np.concatenate((datatrain,datatest), axis=0)
X = data[:,:-1]
Y = data[:,-1]
m = np.shape(X)[0]
n = np.shape(X)[1]

#Do principal component analysis on X
Xnorm,_,_ = normalizar(X)
U,S = pca(Xnorm)
Xpca = Xnorm.dot(U)

#Do principal component analysis on the labeled data, then project the unlabeled data onto those components
datanorm,_,_ = normalizar(data)
U2, S2 = pca(datanorm)
Xpca2 = Xnorm.dot(U2[:-1])

#2D scatter plot of the principal components
fig = scatterPlot(Xpca,Y,(5,9))
plt.title("The first 2 principal components")
fig.show()

#2D scatter plot of the unlabeled data projected onto some components of the labeled data
fig = scatterPlot(Xpca2,Y,(0,20))
plt.title("two functions correlated with the class information")
fig.show()

#3D scatter plot
fig = scatterPlot(Xpca,Y,(5,9,12))
plt.title("The first 3 principal components")
fig.show()

#analyze information loss
totalInfo = sum(S)
infoRate = np.zeros(len(S))
for i in range(len(S)):
    infoRate[i] = np.sum(S[:(i+1)])/totalInfo
plt.figure()
plt.bar(range(len(S)), S/S[0], label="relative weight of each component")
plt.plot(range(len(S)), infoRate, c='r', label="information retained with n components")
plt.legend()
plt.show()

#analyse the correlation of class with the features in each version of the dataset
Ynorm,_,_ = normalizar(Y)
plt.figure()
plt.bar(range(n), Ynorm.dot(Xnorm)/m)
plt.ylim(-1,1)
plt.title("Correlation of class with the initial features")
plt.show()
plt.figure()
plt.bar(range(n), Ynorm.dot(Xpca)/m)
plt.title("Correlation of class with principal components of the unlabeled data")
plt.ylim(-1,1)
plt.show()
plt.figure()
plt.bar(range(n+1),Ynorm.dot(Xpca2)/m)
plt.title("Correlation of class with principal components of the labeled data")
plt.ylim(-1,1)
plt.show()