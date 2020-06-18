from data import *
import matplotlib.pyplot as plt

data = carga_csv("data/train.csv")
X = data[:,:-1]
Y = data[:,-1]

#Do principal component anylisis on X
Xnorm,_,_ = normalizar(X)
U,S = pca(Xnorm)
Xpca = Xnorm.dot(U)

#2D scatter plot7
fig = scatterPlot(Xpca,Y,(0,1))
plt.title("The first 2 principal components")
fig.show()

# #3D scatter plot
# fig = scatterPlot(Xpca,Y,(0,1,2))
# plt.title("The first 3 principal components")
# fig.show()

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