"""
Coursework 4: Neural network training
"""

#%% Imports and definitions
"""
Imports and definitions
"""

from data import carga_csv, normalizar, learningcurve, f1score_multi, accuracy
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
ylimin = -0.02
ylimax = 1.02

#%% Testing the algoritms on a simple hypothesis
"""
2 - Testing the algoritms on a simple hypothesis
"""

reg = 0 # regularization factor
h = 4 # size of the hidden layer

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
3 - Seaching for the best hypothesis with a single split
"""

