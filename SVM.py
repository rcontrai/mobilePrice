"""
Coursework 6: Support Vector Machines
"""

#%%
"""
Imports and definitions
"""

from data import carga_csv,accuracy,f1score_multi
import numpy as np
from sklearn.svm import SVC

    
#%%
"""
1 - Support vector machine
"""


# loading the data
data = carga_csv("data/train.csv")

np.random.seed(0) # To get consistent results while debugging
np.random.shuffle(data)

# Dividing the set into training, validation and test sets
m = np.shape(data)[0]
train = int(np.floor(0.6*m))
val = int(train + np.floor(0.3*m))

Xtrain = data[0:train,:-1]
ytrain = data[0:train,-1]
Xval = data[train:val,:-1]
yval = data[train:val,-1]
Xtest = data[val::,:-1]
ytest = data[val::,-1]

mtrain = np.shape(Xtrain)[0]
mval = np.shape(Xval)[0]
mtest = np.shape(Xtest)[0]

K = 4  # number of classes
C = 100.

# Trainig the model with a linear kernel
svm = SVC(kernel='linear', C=C)
svm.fit(Xtrain,ytrain)

pred = svm.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('SVM linear with C = {0}. Accuracy: {1:1.3g}, F1: {2:1.3g}'.format(C,e,f1))

sigma = 0.1

svm = SVC(kernel='rbf', C=C, gamma = 1/(2*sigma**2))
svm.fit(Xtrain,ytrain)

pred = svm.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('SVM gaussian with C = {0},sigma = {1}. Accuracy: {2:1.3g}, F1: {3:1.3g}'
      .format(C,sigma,e,f1))


#%%
"""
2 - Choice of C and sigma
"""

# Looking for the best values for C and sigma
hpvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

accuracyval = 0
accuracyvalg = 0
f1val = 0
f1valg = 0

for C in hpvalues :
    
    svm = SVC(kernel='linear', C=C)
    svm.fit(Xtrain,ytrain)
    
    pred = svm.predict(Xval)
    prec = f1score_multi(pred,yval,K)
    # prec = accuracy(pred,ytest)
    
    # Choosing the parameters with the smallest validation error
    if (prec > f1val) : 
        accuracyval = accuracy(pred,yval)
        f1val = prec
        Cmin = C
        svmval = svm
    
    for sigma in hpvalues :
        svm = SVC(kernel='rbf', C=C, gamma = 1/(2*sigma**2))
        svm.fit(Xtrain,ytrain)
        
        pred = svm.predict(Xval)
        prec = f1score_multi(pred,yval,K)
        # prec = accuracy(pred,ytest)
        
        # Choosing the parameters with the smallest validation error
        if (prec > f1valg) : 
            accuracyvalg = accuracy(pred,yval)
            f1valg = prec
            Cming = C
            sigmaming = sigma
            svmvalg = svm
   
print("For linear kernel:")         
print('Results on validation SVM linear with optimum found C = {0}.'.format(Cmin) +
      'Accuracy: {0:1.3g}, F1: {1:1.3g}'.format(accuracyval,f1val))

pred = svmval.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('SVM linear on test with C = {0}. Accuracy: {1:1.3g}, F1: {2:1.3g}\n'.format(Cmin,e,f1))


print("For gaussian kernel:")         
print('Results on validation SVM gaussian with optimum founnd C = {0}, sigma = {1}.'.format(Cming,sigmaming)+
      'Accuracy: {0:1.3g}, F1: {1:1.3g}'.format(accuracyvalg,f1valg))

pred = svmvalg.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('SVM gaussian on test with C = {0},sigma = {1}. Accuracy: {2:1.3g}, F1: {3:1.3g}'
      .format(Cming,sigmaming,e,f1))
