"""
Coursework 3: Multi-class logistical regression and neural networks
"""

#%%
"""
Imports and definitions
"""

from data import carga_csv, normalizar, polynomial,accuracy,f1score_multi, kfolds, learningcurve
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    """
    sigmoid function
    can be applied on numbers and on numpy arrays of any dimensions
    """
    return 1 / (1 + np.exp(-z))


def coste_reg(theta, X, Y, lamb):
        """
        cost function with regularization
        computes J(theta) for a given dataset 
        with a regularization factor of lamb
        """
        m = np.shape(X)[0]
        H = sigmoid((np.dot(X,theta)))
        J = -1/m * ( np.log(H).transpose().dot(Y)
                     + np.log(1-H).transpose().dot(1-Y)) 
        reg = lamb/(2*m)*np.sum(theta[1:]**2)
        return J + reg
    
def gradiente_reg(theta, X, Y, lamb):
    """
    gradient function with regularization
    computes the gradient of J at a given theta for a given dataset 
    with a regularization factor of lamb
    """
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = sigmoid(np.dot(X,theta))
    G = 1/m * X.transpose().dot(H - Y)
    reg = np.ones(n,)
    reg[0] = 0
    reg = (lamb/m)*reg*theta
    return G + reg

class MulticlassLogistic:
    
    Th = [];

    
    def predict(self,Xtest):
        """
        computes the prediction for the model theta and examples Xtest
        """
        mtest = np.shape(Xtest)[0]
        Xtest = np.hstack([np.ones([mtest,1]),Xtest])
        
        H = sigmoid(np.dot(Xtest,self.Th.T))
        return np.argmax(H,axis = 1)
    
    
    def fit(self,X,y,num_etiquetas,reg):
        """
        one-vs-all logistical regression
        trains num_etiquetas logistical classifiers to classify as many classes
        uses dataset X and true target variable y
        uses regularization with factor reg
        returns a matrix containing each models' parameters in its lines
        """
        m = np.shape(X)[0]
        n = np.shape(X)[1]
        
        X = np.hstack([np.ones([m,1]),X])
        
        Th = np.zeros((num_etiquetas, n+1))
        # For each class
        for i in range(num_etiquetas):
            # Mark the points as belonging to the class or not
            labels = np.zeros((m,1))
            labels[y == i] = 1  # the class at index 0 has value 1
            labels = labels.ravel()
            
            theta = np.zeros(n+1,)
            # Compute the optimal value for theta
            result = opt.fmin_tnc(func=coste_reg, x0=theta, fprime=gradiente_reg,
                                            args=(X,labels,reg), messages = 0)
            Th[i,:] = result[0] 
            
        self.Th = Th



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


# Object used
ML = MulticlassLogistic();

# Lambdas needed
predict = lambda x: ML.predict(x) 
error = lambda p,y: 1 - f1score_multi(p,y,K)

# For plotting 
ylimin = -0.02
ylimax = 0.62

#%%
"""
2 - Multiclass Logistic 
"""

reg = 0 # regularization factor


# Plot learning curve.
batch = 10;
fit = lambda x,y: ML.fit(x,y,K,reg)
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.title('Learning curve for logistic regression, no regularization')
plt.ylim(ylimin,ylimax)
fig.show()

# Computing precision
ML.fit(Xtrain,ytrain,K,reg)
pred = ML.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Logistical model without regularization accuracy: {0:1.3g}, F1: {1:1.3g}'.format(e,f1))

#%%
"""
3A - Choose lambda with single split
"""

# Train the model for a range of values of lambda
# and compute its error on the training set and on the
# validation set

step = 10;
lamb_arr = range(0,1000,step)
lpts = np.size(lamb_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for lamb in lamb_arr:

    ML.fit(Xt_notval,yt_notval,K,lamb)
       
    pred = ML.predict(Xt_notval)
    prec[i] = error(pred,yt_notval)
    
    pred = ML.predict(Xt_val)
    precval[i] = error(pred,yt_val)
    i = i+1;
    
# Display the lambda curve
plt.figure()
plt.plot(lamb_arr, prec, label = "Train", alpha = 0.5)
plt.plot(lamb_arr, precval, label = "Validation")
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Error') 
plt.ylim(0.,1)
plt.title(r'Selecting $\lambda$ using a validation set')
plt.show()

bestlamb = np.argmin(precval)*step

print('Best lambda with single split = : {}'.format(bestlamb))

#%%
"""
4A - Model with best lambda with single split
"""

reg = bestlamb # regularization factor

# Learning curve
fit = lambda x,y: ML.fit(x,y,K,reg)
batch = 10
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.ylim(ylimin,ylimax)
plt.title('Learning curve for logistic regression with lambda = ' + str(reg))
fig.show()


# Computing precision
ML.fit(Xtrain,ytrain,K,reg)
pred = ML.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Logistical model accuracy with single split lambda = {0}: {1:1.3g}, F1: {2:1.3g}'.format(reg,e,f1))

#%%
"""
3B - Choose lambda with k-fold crossvalidation
"""


k = 5; # number of folds
step = 10;
lamb_arr = range(0,1000,step)
lpts = np.size(lamb_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for lamb in lamb_arr:

    fit = lambda x,y: ML.fit(x,y,K,lamb)
    
    prec[i],precval[i] = kfolds(Xtrain,ytrain,fit,predict,error,k)
    i = i +1;
    
    
# Display the lambda curves
plt.figure()
plt.plot(lamb_arr, prec, label = "Train", alpha = 0.5)
plt.plot(lamb_arr, precval, label = "Average validation")
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Error') 
plt.ylim(0,1)
plt.title(r'Selecting $\lambda$ using k-folds')
plt.show()

bestlambkfolds = np.argmin(precval)*step

print('Best lambda with kfolds = : {}'.format(bestlambkfolds))


#%%
"""
4B - Model with best lambda with kfolds
"""

reg = bestlambkfolds # regularization factor

# Learning curve
fit = lambda x,y: ML.fit(x,y,K,reg)
batch = 10
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.ylim(ylimin,ylimax)
plt.title('Learning curve for logistic regression with lambda = ' + str(reg))
fig.show()


# Computing precision
ML.fit(Xtrain,ytrain,K,reg)
pred = ML.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Logistical model accuracy with kfolds lambda = {0}: {1:1.3g}, F1: {2:1.3g}'.format(reg,e,f1))

#%%
"""
5 - Polynomial features d = 2 + high lambda = 10000
"""

batch = 10;
reg = 0;

d = 2;
inter_only = False

# Polynomial transformations
Xtrain_pol = polynomial(Xtrain,d, inter_only)
Xtest_pol = polynomial(Xtest,d, inter_only)

# Normalize
Xtrain_pol, muTrain_pol, sigmaTrain_pol = normalizar(Xtrain_pol)
Xtest_pol = normalizar(Xtest_pol, muTrain_pol, sigmaTrain_pol)

# Single split in validation and not validation
Xt_notval_pol = Xtrain_pol[0:split,:]
Xt_val_pol = Xtrain_pol[split::,:]

# For plotting 
ylimin = -0.02
ylimax = 0.7

# Learning curve
fit = lambda x,y: ML.fit(x,y,K,reg)
fig = learningcurve(Xt_notval_pol,yt_notval,Xt_val_pol,yt_val,fit,predict,error,batch)
plt.title('Learning curve for logistic regression with d = {0} without regularisation'.format(d))
fig.show()


# Computing precision
ML.fit(Xtrain_pol,ytrain,K,reg)
pred = ML.predict(Xtest_pol)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Logistical model with d = {0} without regularisation accuracy: {1:1.3g}, F1: {2:1.3g}'.format(d,e,f1))

#%%
"""
6A - Choose lambda with single split
"""

# Train the model for a range of values of lambda
# and compute its error on the training set and on the
# validation set

step = 10;
lamb_arr = range(0,1000,step)
lpts = np.size(lamb_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for lamb in lamb_arr:

    ML.fit(Xt_notval_pol,yt_notval,K,lamb)
       
    pred = ML.predict(Xt_notval_pol)
    prec[i] = error(pred,yt_notval)
    
    pred = ML.predict(Xt_val_pol)
    precval[i] = error(pred,yt_val)
    i = i+1;
    
# Display the lambda curve
plt.figure()
plt.plot(lamb_arr, prec, label = "Train", alpha = 0.5)
plt.plot(lamb_arr, precval, label = "Validation")
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Error') 
plt.ylim(0.,1)
plt.title(r'Selecting $\lambda$ using a validation set')
plt.show()

bestlamb = np.argmin(precval)*step

print('Best lambda with single split = : {}'.format(bestlamb))

#%%
"""
7A - Model with best lambda with single split
"""

reg = bestlamb # regularization factor

# Learning curve
fit = lambda x,y: ML.fit(x,y,K,reg)
batch = 10
fig = learningcurve(Xt_notval_pol,yt_notval,Xt_val_pol,yt_val,fit,predict,error,batch)
plt.ylim(ylimin,ylimax)
plt.title('Learning curve for logistic regression with d ={0}, lambda = {1}'.format(d,reg))
fig.show()


# Computing precision
ML.fit(Xtrain_pol,ytrain,K,reg)
pred = ML.predict(Xtest_pol)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Logistical model accuracy with d = {0} and single split lambda = {1}:'.format(d,reg) +
      '{0:1.3g}, F1: {1:1.3g}'.format(e,f1))

#%%
"""
6B - Choose lambda with k-fold crossvalidation
"""


k = 5; # number of folds
step = 10;
lamb_arr = range(0,1000,step)
lpts = np.size(lamb_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for lamb in lamb_arr:

    fit = lambda x,y: ML.fit(x,y,K,lamb)
    
    prec[i],precval[i] = kfolds(Xtrain_pol,ytrain,fit,predict,error,k)
    i = i +1;
    
    
# Display the lambda curves
plt.figure()
plt.plot(lamb_arr, prec, label = "Train", alpha = 0.5)
plt.plot(lamb_arr, precval, label = "Average validation")
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Error') 
plt.ylim(0,1)
plt.title(r'Selecting $\lambda$ using k-folds')
plt.show()

bestlambkfolds = np.argmin(precval)*step

print('Best lambda with kfolds = : {}'.format(bestlambkfolds))


#%%
"""
7B - Model with best lambda with kfolds
"""

reg = bestlambkfolds # regularization factor

# Learning curve
fit = lambda x,y: ML.fit(x,y,K,reg)
batch = 10
fig = learningcurve(Xt_notval_pol,yt_notval,Xt_val_pol,yt_val,fit,predict,error,batch)
plt.ylim(ylimin,ylimax)
plt.title('Learning curve for logistic regression with d ={0}, lambda = {1}'.format(d,reg))
fig.show()


# Computing precision
ML.fit(Xtrain_pol,ytrain,K,reg)
pred = ML.predict(Xtest_pol)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Logistical model accuracy with d = {0} and kfolds lambda = {1}:'.format(d,reg) +
      '{0:1.3g}, F1: {1:1.3g}'.format(e,f1))