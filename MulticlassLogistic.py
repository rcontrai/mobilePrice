"""
Coursework 3: Multi-class logistical regression and neural networks
"""

#%%
"""
Imports and definitions
"""

from data import carga_csv, polynomial
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

def precision(theta,X,Y):
    """
    accuracy function
    computes the accuracy of the logistic model theta on X with true target variable Y
    """
    m = np.shape(X)[0]
    H = sigmoid(np.dot(X,theta.T))
    labels = np.argmax(H,axis = 1)
    Y = Y.ravel()
    return np.sum(labels == Y)/m


def oneVsAll(X,y,num_etiquetas,reg):
    """
    one-vs-all logistical regression
    trains num_etiquetas logistical classifiers to classify as many classes
    uses dataset X and true target variable y
    uses regularization with factor reg
    returns a matrix containing each models' parameters in its lines
    """
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    
    Th = np.zeros((num_etiquetas, n))
    # For each class
    for i in range(num_etiquetas):
        # Mark the points as belonging to the class or not
        labels = np.zeros((m,1))
        labels[y == i] = 1  # the class at index 0 has value 1
        labels = labels.ravel()
        
        theta = np.zeros(n,)
        # Compute the optimal value for theta
        result = opt.fmin_tnc(func=coste_reg, x0=theta, fprime=gradiente_reg,
                                        args=(X,labels,reg), messages = 0)
        Th[i,:] = result[0] 
        
    return Th



#%% 
"""
1 - Multi-class logistical regression (lambda = 0, 0.815)
"""

# loading the data
data = carga_csv("data/train.csv")

np.random.seed(0) # To get consistent results while debugging
np.random.shuffle(data)

# Dividing the set into training, validation and test sets
m = np.shape(data)[0]
train = int(np.floor(0.8*m))
val = int(train + np.floor(0.1*m))

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
reg = 0 # regularization factor

# Training the models
Xtrainp = np.hstack([np.ones([mtrain,1]),Xtrain])
Th = oneVsAll(Xtrainp,ytrain,K,reg)

# Computing precision

Xtestp = np.hstack([np.ones([mtest,1]),Xtest])
p = precision(Th,Xtestp,ytest)
print('Logistical model accuracy : {0:1.3g}'.format(p))

#%%
"""
2 - Plot bias-variance plot
"""

batch = 10;
prec = np.zeros(int(mtrain/batch)-1,)
precval = np.zeros(int(mtrain/batch)-1,)

Xvalp = np.hstack([np.ones([mval,1]),Xval])

for i in range(1,int(mtrain/batch)):
    Xi = Xtrainp[0:i*batch]
    yi = ytrain[0:i*batch]
    
    Th = oneVsAll(Xi,yi,K,reg)
        
    prec[i-1] = 1-precision(Th,Xi,yi)
    precval[i-1] = 1-precision(Th,Xvalp,yval)
    

# Display the learning curves
plt.figure()
plt.plot(prec, label = "Train")
plt.plot(precval, label = "Cross validation")
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Learning curve for logistic regression')
plt.show()


#%%
"""
3 - High variance - Lambda
"""

# Train the model for a range of values of lambda
# and compute its error on the training set and on the
# validation set

lamb_arr = range(0,1000,10)
lpts = np.size(lamb_arr)

prec = np.zeros(lpts,)
precval = np.zeros(lpts,)

i = 0;
for lamb in lamb_arr:

    Th = oneVsAll(Xtrainp,ytrain,K,lamb)
        
    prec[i] = 1-precision(Th,Xtrainp,ytrain)
    precval[i] = 1-precision(Th,Xvalp,yval)
    i = i+1;
    
# Display the learning curves
plt.figure()
plt.plot(lamb_arr, prec, label = "Train")
plt.plot(lamb_arr, precval, label = "Cross validation")
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Error') 
plt.title(r'Selecting $\lambda$ using a cross-validation set')
plt.show()

print('Best lambda = : {}'.format(np.argmin(precval) * 10))


#%%
"""
4 - Model with lambda = 810 (0.885)
"""
K = 4  # number of classes
reg = 810 # regularization factor

# Training the models
Th = oneVsAll(Xtrainp,ytrain,K,reg)

# Computing precision

Xtestp = np.hstack([np.ones([mtest,1]),Xtest])
p = precision(Th,Xtestp,ytest)
print('Logistical model accuracy : {0:1.3g}'.format(p))


batch = 10;
prec = np.zeros(int(mtrain/batch)-1,)
precval = np.zeros(int(mtrain/batch)-1,)

for i in range(1,int(mtrain/batch)):
    Xi = Xtrainp[0:i*batch]
    yi = ytrain[0:i*batch]
    
    Th = oneVsAll(Xi,yi,K,reg)
        
    prec[i-1] = 1-precision(Th,Xi,yi)
    precval[i-1] = 1-precision(Th,Xvalp,yval)
    

# Display the learning curves
plt.figure()
plt.plot(prec, label = "Train")
plt.plot(precval, label = "Cross validation")
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Learning curve for logistic regression')
plt.show()

#%%
"""
5 - Polynomial features d = 2 + high lambda = 10000
"""

batch = 10;
reg = 10000;

d = 2;

prec = np.zeros(int(mtrain/batch)-1,)
precval = np.zeros(int(mtrain/batch)-1,)

Xtrain_pol, mu, sigma = polynomial(Xtrain,d)
Xval_pol, _, _ = polynomial(Xval,d)

Xval_pol[:,1:] = (Xval_pol[:,1:] - mu) / sigma

for i in range(1,int(mtrain/batch)):
    Xi = Xtrain_pol[0:i*batch]
    yi = ytrain[0:i*batch]
    
    Th = oneVsAll(Xi,yi,K,reg)
    
    
    prec[i-1] = 1-precision(Th,Xi,yi)
    precval[i-1] = 1-precision(Th,Xval_pol,yval)
    

# Display the learning curves
plt.figure()
plt.plot(prec, label = "Train")
plt.plot(precval, label = "Cross validation")
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Learning curve for logistic regression')
plt.show()

# Computing precision

Xtest_pol, _, _ = polynomial(Xtest,d)
Xtest_pol[:,1:] = (Xtest_pol[:,1:] - mu) / sigma
p = precision(Th,Xtest_pol,ytest)
print('Logistical model accuracy : {0:1.3g}'.format(p))