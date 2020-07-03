"""
Coursework 4: Neural network training
"""

#%% Imports and definitions
"""
Imports and definitions
"""

from data import carga_csv, normalizar, f1score_multi, accuracy, learningcurve, kfolds
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    """
    sigmoid function
    can be applied on numbers and on numpy arrays of any dimensions
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    """
    derivative of a sigmoid function
    can be applied on numbers and on numpy arrays of any dimensions
    """
    gz = sigmoid(z)
    return gz*(1-gz)

def propagation(theta1,theta2,X):
    """
    forward propagation function
    performs the forward propagation in a neural network with the stucture decribed in the subject,
    with weights given by theta1 and theta2
    on the dataset X
    returns :
        -H : the activation of the output layer
        -a2 : the activation of the hidden layer
    """
    m = np.shape(X)[0]
    z2 = (np.dot(X,theta1.T))
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones([m,1]),a2])
    H = sigmoid(np.dot(a2,theta2.T))
    return H, a2

def coste_reg(theta1, theta2, X, Y, lamb):
    """
    cost function with regularization
    computes J for a given neural network,
    described by the weights matrices theta1 and theta2
    on a given dataset, described by X and Y
    with a regularization factor of lamb
    """
    m = np.shape(X)[0]
    H, _ = propagation(theta1,theta2, X)
    J = -1/m * np.sum( np.log(H)*Y
                 + np.log(1-H)*(1-Y)) 
    reg = lamb/(2*m)*(np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2))
    return J + reg

def gradiente_reg(theta1, theta2, X, Y, lamb):
    """
    computes the gradient of the cost function
    using backpropagation
    for a neural network described by theta1 and theta2
    on a given dataset described by X and Y
    with a regularization factor of lamb
    returns :
        -d1 : the gradients for the weights of theta1
        -d2 : the gradient for the weights of theta2
    """
    m = np.shape(X)[0]
    
    a3, a2 = propagation(theta1, theta2, X)
    
    delta3 = a3 - Y
    delta2 = delta3.dot(theta2)*(a2*(1-a2))
    DELTA1 = delta2[:,1:].T.dot(X)
    DELTA2 = delta3.T.dot(a2)
    reg1 = np.zeros_like(theta1)
    reg1[:,1:] = lamb / m * theta1[:,1:]
    reg2 = np.zeros_like(theta2)
    reg2[:,1:] = lamb / m * theta2[:,1:]
    return DELTA1/m + reg1, DELTA2/m + reg2
    
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas , X, y,  reg) :
    """
    computes the cost function and its gradient 
    for a given neural network having 3 layers
    and for a given dataset, while performing
    regularization
    parameters :
        -params_rn : vector of the weights of the network
        -num_entradas : size of the input layer
        -num_ocultas : size of the hidden layer
        -num_etiquetas : size of the output layer
        -X : matrix containing the training examples as vectors in its lines
        -y : matrix containing the labels as vectors in its lines
        -reg : regularization factor
    returns :
        -coste : the value of the cost function
        -grad : the gradient as a vector
    """
    # Add the column of 1s
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m,1]),X])
    
    
    theta1 = np.reshape(params_rn[:num_ocultas*(num_entradas + 1)] ,
            (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1):] ,
            (num_etiquetas, (num_ocultas + 1)))

    coste = coste_reg(theta1, theta2, X, y, reg)
    d1, d2 = gradiente_reg(theta1, theta2, X, y, reg)
    grad = np.concatenate([d1.ravel() , d2.ravel()])
    return coste, grad

def pesosAleatorios(L_in, L_out) :
    """
    returns a matrix of random floats 
    of shape (L_out, L_in + 1)
    the values belong in an interval whose size depends on L_in and L_out
    """
    eps = np.sqrt(6 / (L_in + L_out))
    theta = np.random.rand(L_out, L_in + 1)
    theta = (theta - 0.5)*2*eps
    return theta
    
def precision_neur(theta1,theta2,X,Y):
    """
    accuracy function
    computes the accuracy of the neural network described by theta1 and theta2
    on dataset X with true target variable Y
    """
    m = np.shape(X)[0]
    X_new = np.hstack([np.ones([m,1]),X])
    H,_ = propagation(theta1,theta2,X_new)
    labels = np.argmax(H,axis = 1)
    Y = Y.ravel()
    return np.sum(labels == Y)/m
    
class NeuralNetwork:
    
    theta1 = []
    theta2 = []
    theta1init = [] #We save the initial weights separately so that
    theta2init = [] #successive trainings can be done independantly
    
    def __init__(self, inputSize, hiddenSize, outputSize):
        """
        initializes the two weight matrices with the specified shapes and random weights
        """
        self.theta1 = pesosAleatorios(inputSize, hiddenSize)
        self.theta2 = pesosAleatorios(hiddenSize, outputSize)
        self.theta1init = np.copy(self.theta1)
        self.theta2init = np.copy(self.theta2)
    
    
    def predict(self,Xtest):
        """
        computes the prediction of the model on examples Xtest
        """
        m = np.shape(Xtest)[0]
        Xtest = np.hstack([np.ones([m,1]),Xtest])
        H,_ = propagation(self.theta1,self.theta2,Xtest)
        labels = np.argmax(H,axis = 1)
        return labels
    
    
    def fit(self,X,y,reg):
        maxIter = 2000
        m = np.shape(X)[0]
        
        # Reshape the weights
        num_entradas = np.shape(self.theta1)[1] -1 
        num_ocultas = np.shape(self.theta2)[1] -1
        num_etiquetas = np.shape(self.theta2)[0]
        params_rn = np.concatenate([self.theta1init.ravel() , self.theta2init.ravel()])
        
        # Transform y into a usable matrix
        labels = np.zeros((m,num_etiquetas))
        for i in range(num_etiquetas) :
            labels[(y == i).ravel(), i] = 1;
        
        trainingresult = opt.minimize(fun=backprop, x0=params_rn,
                                    args=(num_entradas, num_ocultas, num_etiquetas, X, labels, reg),
                                    method='TNC', jac=True,
                                    options={'maxiter': maxIter})
        
        params_rn_result = trainingresult.x
        
        # Compute the precision
        self.theta1 = np.reshape(params_rn_result[:num_ocultas*(num_entradas + 1)] ,
                (num_ocultas, (num_entradas + 1)))
        self.theta2 = np.reshape(params_rn_result[num_ocultas*(num_entradas + 1):] ,
                (num_etiquetas, (num_ocultas + 1)))


#%% Load data
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
n = np.shape(Xtrain)[1]

# Split into training and validation set
np.random.seed(0) # To get consistent results while debugging
np.random.shuffle(datatrain)
split = int(np.floor(0.7*mtrain))

Xt_notval = datatrain[0:split,:-1]
yt_notval = datatrain[0:split,-1]
Xt_val = datatrain[split::,:-1]
yt_val = datatrain[split::,-1]


# Lambdas needed
error = lambda p,y: 1 - f1score_multi(p,y,K)

# For plotting 
ylimin = 0   #boundaries for the learning curves
ylimax = 1
lamblimin = 0  #boundaries for the lambda curves
lamblimax = 1
barlimin = 0  # boundaries for the hidden layer size curves
barlimax = 0.07

#%% Testing the algoritms on a simple hypothesis
"""
2 - Testing the algoritms on a simple hypothesis
"""

reg = 0 # regularization factor
h = 4# size of the hidden layer

# Initialize the network
NN = NeuralNetwork(n,h,K)

# Plot learning curve.
batch = 10;
fit = lambda x,y: NN.fit(x,y,reg)
predict = lambda x: NN.predict(x) 
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.title('Learning curve for neural network, {0} hidden neurons, no regularization'.format(h))
plt.ylim(ylimin,ylimax)
fig.show()

# Computing precision
NN.fit(Xtrain,ytrain,reg)
pred = NN.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Neural network with {0} hidden neurons and no regularization accuracy: {1:1.3g}, F1: {2:1.3g}'.format(h, e,f1))


#%% Seaching for the best hypothesis with a single split
"""
3A - Seaching for the best hypothesis with a single split
"""

# Train models for a range of values of lambda and h
# and compute their error on the training set and on the
# validation set

lamb_arr = np.append(0,np.logspace(0,2.35,num=30))
h_list = [1,2,4,8,12,16] #range of sizes for the hidden layer
nmodels = len(h_list)
nn_list = [] #list of neural networks with different architectures
for h in h_list:
    nn_list.append(NeuralNetwork(n,h,K))
    
prec = np.zeros((nmodels,lamb_arr.shape[0])) #model precision on the training set
precval = np.zeros_like(prec)   #model precision on the validation set

j = 0
for NN in nn_list:
    i=0
    for lamb in lamb_arr:
        NN.fit(Xt_notval,yt_notval,lamb)
        
        pred = NN.predict(Xt_notval)
        prec[j][i] = error(pred,yt_notval)
        
        pred = NN.predict(Xt_val)
        precval[j][i] = error(pred,yt_val)
        i += 1
    j += 1

#get the results
bestIndices = np.argmin(precval,axis=1)
bestPrecvals = np.zeros(nmodels)
bestPrecs = np.zeros(nmodels)
for j in range(nmodels):
    bestPrecvals[j] = precval[j][bestIndices[j]]
    bestPrecs[j] = prec[j][bestIndices[j]]

bestJ = np.argmin(bestPrecvals) # index of the best model
besth = h_list[bestJ]
bestNN = nn_list[bestJ]
bestlamb = lamb_arr[bestIndices][bestJ]

# Display the lambda curve for the best hypothesis
plt.figure()
plt.plot(lamb_arr, prec[bestJ], label = "Train", alpha = 0.8)
plt.plot(lamb_arr, precval[bestJ], label = "Validation")
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Error') 
plt.ylim(lamblimin,lamblimax)
plt.title(r'Selecting $\lambda$ using a validation set for h=' + str(besth))
plt.show()

# Compare the results of the hypotheses
plt.figure()
plt.bar(h_list, bestPrecs, label = "Train", width = -0.4, align = 'edge')
plt.bar(h_list, bestPrecvals, label = "Validation", width = 0.4, align = 'edge')
plt.legend()
plt.xlabel('Size of hidden layer')
plt.ylabel('Best error')
plt.ylim(barlimin,barlimax)
plt.title('Selecting h using a validation set')
plt.show()

print('Best lambda with a single split = : {}'.format(bestlamb))
print('Best number of hidden neurons with a single split = : {}'.format(besth))

#%% Model with best hypothesis and lambda found with a single split
"""
3B - Model with best hypothesis and lambda found with a single split
"""

reg = bestlamb # regularization factor

# Learning curve
fit = lambda x,y: bestNN.fit(x,y,reg)
predict = lambda x: bestNN.predict(x) 
batch = 10
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.ylim(ylimin,ylimax)
plt.title('Learning curve for neural network, {0} hidden neurons, lambda = {1:1.3g}'.format(h,reg))
fig.show()


# Computing precision
bestNN.fit(Xtrain,ytrain,reg)
pred = bestNN.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Neural network accuracy with single split lambda = {0:1.3g}, h = {3}: {1:1.3g}, F1: {2:1.3g}'.format(reg,e,f1,besth))


#%% Seaching for the best hypothesis with k-fold crossvalidation
"""
3A - Seaching for the best hypothesis with k-fold crossvalidation
"""

# Train models for a range of values of lambda and h
# and compute their average error on several pairs of
# training and validation sets

k = 5 # number of folds

lamb_arr = np.append(0,np.logspace(0,2.35,num=30))
h_list = [1,2,4,8,12,16] #range of sizes for the hidden layer
nmodels = len(h_list)
nn_list = [] #list of neural networks with different architectures
for h in h_list:
    nn_list.append(NeuralNetwork(n,h,K))
    
prec = np.zeros((nmodels,lamb_arr.shape[0])) #model precision on the training set
precval = np.zeros_like(prec)   #model precision on the validation set

j = 0
for NN in nn_list:
    i=0
    for lamb in lamb_arr:
        fit = lambda x,y : NN.fit(x,y,lamb)
        predict = lambda x : NN.predict(x)
        prec[j][i], precval[j][i] = kfolds(Xtrain,ytrain,fit,predict,error,k)
        i += 1
    j += 1

#get the results
bestIndices = np.argmin(precval,axis=1)
bestPrecvals = np.zeros(nmodels)
bestPrecs = np.zeros(nmodels)
for j in range(nmodels):
    bestPrecvals[j] = precval[j][bestIndices[j]]
    bestPrecs[j] = prec[j][bestIndices[j]]

bestJ = np.argmin(bestPrecvals) # index of the best hypothesis
besth = h_list[bestJ]
bestNN = nn_list[bestJ]
bestlamb = lamb_arr[bestIndices][bestJ]

# Display the lambda curve for the best hypothesis
plt.figure()
plt.plot(lamb_arr, prec[bestJ], label = "Train", alpha = 0.8)
plt.plot(lamb_arr, precval[bestJ], label = "Average validation")
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Error') 
plt.ylim(lamblimin,lamblimax)
plt.title(r'Selecting $\lambda$ using k-fold crossvalidation for h=' + str(besth))
plt.show()

# Compare the results of the hypotheses
plt.figure()
plt.bar(h_list, bestPrecs, label = "Train", width = -0.4, align = 'edge')
plt.bar(h_list, bestPrecvals, label = "Average validation", width = 0.4, align = 'edge')
plt.legend()
plt.xlabel('Size of hidden layer')
plt.ylabel('Best error')
plt.ylim(barlimin,barlimax)
plt.title('Selecting h using k-fold crossvalidation')
plt.show()

print('Best lambda with kfolds = : {}'.format(bestlamb))
print('Best number of hidden neurons with kfolds = : {}'.format(besth))

#%% Model with best hypothesis and lambda found with kfolds
"""
3B - Model with best hypothesis and lambda found with kfolds
"""

reg = bestlamb # regularization factor

# Learning curve
fit = lambda x,y: bestNN.fit(x,y,reg)
predict = lambda x: bestNN.predict(x) 
batch = 10
fig = learningcurve(Xt_notval,yt_notval,Xt_val,yt_val,fit,predict,error,batch)
plt.ylim(ylimin,ylimax)
plt.title('Learning curve for neural network, {0} hidden neurons, lambda = {1:1.2g}'.format(h,reg))
fig.show()


# Computing precision
bestNN.fit(Xtrain,ytrain,reg)
pred = bestNN.predict(Xtest)
f1 = f1score_multi(pred,ytest,K)
e = accuracy(pred,ytest)
print('Neural network accuracy with with kfolds lambda = {0:1.3g}, h = {3}: {1:1.3g}, F1: {2:1.3g}'.format(reg,e,f1,besth))