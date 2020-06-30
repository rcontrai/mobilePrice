"""
Random forest
"""

#%%
"""
Imports and definitions
"""

from data import carga_csv, accuracy,f1score_multi, kfolds, learningcurve, normalizar
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt

#%% 
"""
1 - Load data
"""

K = 4  # number of classes
# loading the data
datatrain = carga_csv("data/train.csv")
datatest = carga_csv("data/test.csv")

#Normalization
datatrain[:,:-1], muTrain, sigmaTrain = normalizar( datatrain[:,:-1])
datatest[:,:-1] = normalizar(datatest[:,:-1], muTrain, sigmaTrain)

Xtrain = datatrain[:,:-1]
ytrain = datatrain[:,-1]
Xtest = datatest[:,:-1]
ytest = datatest[:,-1]

mtrain = np.shape(Xtrain)[0]
mtest = np.shape(Xtest)[0]

# Split into training and validation set
np.random.seed(0) # To get consistent results while debugging
np.random.shuffle(datatrain)
split = int(np.floor(0.7*mtrain))

Xt_notval = datatrain[0:split,:-1]
yt_notval = datatrain[0:split,-1]
Xt_val = datatrain[split::,:-1]
yt_val = datatrain[split::,-1]


# Lambdas needed
fit = lambda x,y: rfc.fit(x,y)
predict = lambda x: rfc.predict(x)
error = lambda p,y: 1 - f1score_multi(p,y,K)

# For plotting 
ylimin = -0.02
ylimax = 0.71

#%%
"""
2A - Choose #estimators with single split
"""

# Train the model for a range of values for n_estimators
# and compute its error on the training set and on the
# validation set

est_arr = range(1,300,10)
lpts = np.size(est_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for est in est_arr:
    
    rfc = RFC(n_estimators=est)
    rfc.fit(Xtrain,ytrain)
       
    pred = rfc.predict(Xt_notval)
    prec[i] = error(pred,yt_notval)
    
    pred = rfc.predict(Xt_val)
    precval[i] = error(pred,yt_val)
    i = i+1;
    
# Display the lambda curve
plt.figure()
plt.plot(est_arr, prec, label = "Train", alpha = 0.5)
plt.plot(est_arr, precval, label = "Validation")
plt.legend()
plt.xlabel('#estimators')
plt.ylabel('Error') 
plt.title(r'Selecting #estimators using a validation set')
plt.show()

bestest = est_arr[np.argmin(precval)]

print('Best #estimators with single split = : {}'.format(bestest))

#%%
"""
3A -  Model with best #estimators with single split
"""

est = bestest # regularization factor
rfc = RFC(n_estimators=est)

# Learning curve
batch = 10;
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.title('Random Forest with #estimators = ' + str(est))
plt.ylim(ylimin,ylimax)
fig.show()


# Computing precision
rfc.fit(Xtrain,ytrain)
pred = rfc.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Random forest accuracy with single split #estimators = {0}:'.format(est) + 
      '{0:1.3g}, F1: {1:1.3g}'.format(e,f1))

#%%
"""
2B - Choose #estimators with k-fold crossvalidation
"""


k = 5; # number of folds
est_arr = range(1,300,10)
lpts = np.size(est_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for est in est_arr:
    
    rfc = RFC(n_estimators=est)
    
    prec[i],precval[i] = kfolds(Xtrain,ytrain,fit,predict,error,k)
    i = i +1;
    
    
# Display the lambda curves
plt.figure()
plt.plot(est_arr, prec, label = "Train", alpha = 0.5)
plt.plot(est_arr, precval, label = "Average validation")
plt.legend()
plt.xlabel('#estimators')
plt.ylabel('Error') 
plt.title(r'Selecting #estimators using k-folds')
plt.show()

bestestkfolds = est_arr[np.argmin(precval)]

print('Best #estimators with kfolds = : {}'.format(bestestkfolds))


#%%
"""
3B - Model with best #estimators with kfolds
"""

est = bestestkfolds # regularization factor
rfc = RFC(n_estimators=est)

# Learning curve
batch = 10;
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.title('Random Forest with #estimators = ' + str(est))
plt.ylim(ylimin,ylimax)
fig.show()


# Computing precision
rfc.fit(Xtrain,ytrain)
pred = rfc.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Random forest accuracy with kfolds #estimators = {0}:'.format(est) + 
      '{0:1.3g}, F1: {1:1.3g}'.format(e,f1))
