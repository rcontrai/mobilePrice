"""
Coursework 6: Support Vector Machines
"""

#%%
"""
Imports and definitions
"""

from data import carga_csv, accuracy,f1score_multi, kfolds, learningcurve
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#%% 
"""
1 - Load data
"""

K = 4  # number of classes
# loading the data
datatrain = carga_csv("data/train.csv")
datatest = carga_csv("data/test.csv")

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
fit = lambda x,y: svm.fit(x,y)
predict = lambda x: svm.predict(x)
error = lambda p,y: 1 - f1score_multi(p,y,K)

#%%
"""
2 - Linear SVM 
"""

C = 1 # regularization factor
svm = SVC(kernel='linear', C=C)

# Plot learning curve.
batch = 100;
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.title('Learning curve for linear SVM, no regularization')
fig.show()

# Computing precision
svm.fit(Xtrain,ytrain)
pred = svm.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Linear SVM without regularization accuracy: {0:1.3g}, F1: {1:1.3g}'.format(e,f1))

#%%
"""
3A - Linear - Choose C with single split
"""

# Train the model for a range of values of lambda
# and compute its error on the training set and on the
# validation set

C_arr = [0.1, 0.5, 1, 10, 100, 1000, 5000, 10000]
lpts = np.size(C_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for C in C_arr:
    
    svm = SVC(kernel='linear', C=C)
    svm.fit(Xt_notval,yt_notval)
       
    pred = svm.predict(Xt_notval)
    prec[i-1] = error(pred,yt_notval)
    
    pred = svm.predict(Xt_val)
    precval[i-1] = error(pred,yt_val)
    i = i+1;
    
# Display the lambda curve
plt.figure()
plt.plot(C_arr, prec, label = "Train", alpha = 0.5)
plt.plot(C_arr, precval, label = "Validation")
plt.legend()
plt.xlabel('$C$')
plt.ylabel('Error') 
plt.title(r'Selecting $C$ using a validation set')
plt.show()

bestC = C_arr[np.argmin(precval)]

print('Best C with single split = : {}'.format(bestC))

#%%
"""
4A - Linear - Model with best C with single split
"""

C = bestC # regularization factor
svm = SVC(kernel='linear', C=C)

# Learning curve
batch = 100;
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.title('Linear SVM with C = ' + str(C))
fig.show()


# Computing precision
svm.fit(Xtrain,ytrain)
pred = svm.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Linear SVM accuracy with single split C = {0}: {1:1.3g}, F1: {2:1.3g}'.format(C,e,f1))

#%%
"""
3B - Linear - Choose C with k-fold crossvalidation
"""


k = 5; # number of folds
C_arr = [0.1, 0.5, 1, 5, 10, 100, 1000, 5000, 10000]
lpts = np.size(C_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for C in C_arr:
    
    svm = SVC(kernel='linear', C=C)
    
    prec[i],precval[i] = kfolds(Xtrain,ytrain,fit,predict,error,k)
    i = i +1;
    
    
# Display the lambda curves
plt.figure()
plt.plot(C_arr, prec, label = "Train", alpha = 0.5)
plt.plot(C_arr, precval, label = "Average validation")
plt.legend()
plt.xlabel('$C$')
plt.ylabel('Error') 
plt.title(r'Selecting $C$ using k-folds')
plt.show()

bestCkfolds = C_arr[np.argmin(precval)]

print('Best C with kfolds = : {}'.format(bestCkfolds))


#%%
"""
4B - Linear - Model with best lambda with kfolds
"""

C = bestCkfolds # regularization factor
svm = SVC(kernel='linear', C=C)

# Learning curve
batch = 100;
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.title('Linear SVM with C = ' + str(C))
fig.show()


# Computing precision
svm.fit(Xtrain,ytrain)
pred = svm.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Linear SVM accuracy with kfolds C = {0}: {1:1.3g}, F1: {2:1.3g}'.format(C,e,f1))




#%%
"""
5 - Other kernels SVM 
"""

C = 1 # regularization factor
gamma = 0.05 # gamma factor
kernel = 'rbf' # kernel
svm = SVC(kernel=kernel, C=C, gamma = gamma)

# Plot learning curve.
batch = 100;
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.title('Learning curve for SVM with kernel ' + str(kernel) +
          ' with parameters C = ' + str(C) + ' and gamma = ' + str(gamma))
fig.show()

# Computing precision
svm.fit(Xtrain,ytrain)
pred = svm.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('SVM with kernel ' + str(kernel) + 
      ' with parameters C = {0}, gamma = {1}'.format(C,gamma) +
      ', accuracy: {0:1.3g}, F1: {1:1.3g}'.format(e,f1))
