"""
data.py :
provides tools for loading, preprocessing, visualizing the data
"""

from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#%% loading

def carga_csv(file_name):
    """
    carga el fichero csv especificado y lo
    devuelve en un array de numpy
    """
    #header=0 para que detecta automaticámente los nombres de columnas
    valores = read_csv(file_name,header=0).values 
    #suponemos que siempre trabajaremos con float
    return valores.astype(float)

#%% preprocessing

def normalizar(X,mu=np.array([]),sigma=np.array([])):
    """
    Normalizes the dataset vector X
    and returns the normalized dataset
    if pMu and pSigma are not set,this will also return
    the means and the standard deviantion for each feature
    input:
        X : the array containing vectorized examples in its rows
        mu = np.array([]) : a vector of length n containinig the mean of each feature
        sigma = np.array([]) : a vector of length n containinig the standard deviation of each feature
    """
    if ( (mu.shape[0] == 0) or (sigma.shape[0] == 0)):
        if( mu.shape[0] != sigma.shape[0]):
            print("Warning : one of the optional arguments of normalizar was not set")
        mu = np.mean(X,  axis=0)
        sigma = np.std(X, axis=0)
        X_norm = (X - mu) / sigma
        return X_norm, mu, sigma
    else:
        X_norm = (X - mu) / sigma
        return X_norm
    

def pca(X):
    """
    performs principal component analysis (PCA) on the dataset
    input :
        X : the array containing vectorized examples in its rows,
            normalized
    returns :
        U : the matrix of eigenvectors of the covariance matrix,
            containing eigenvectors in its columns ordered by eigenvalue
        S : an array containing the eigenvalues associated with the eigenvectors
    """
    m = np.shape(X)[0]
    sigma = (1/m) * np.transpose(X).dot(X) #covariance matrix
    U,S,_ = np.linalg.svd(sigma) #singular value decomposition
    return U,S


def polynomial(X,grad,inter_only):
    """
    Computes each power of X's values from 0 to grad
    """
    poly = PolynomialFeatures(grad,interaction_only=inter_only,include_bias=False)
    X_pol = poly.fit_transform(X)
    return X_pol
    

#%% visualization and metrics

def scatterPlot(X,Y,features=(0,1)):
    """
    Plots 2 or 3 features of the dataset in a scatter plot
    input: 
      X : the array containing vectorized examples in its rows
      Y : the array containing the class of each example
      features : a tuple giving the indices of the features to display
                 it can have length 2 or 3
                 a length of 2 will result in a 2d plot
                 a length of 3 will result in a 3d plot
                 default value = (0,1)
    returns:
        a matplotlib.Figure object containing the scatter plot
    """
    fig = plt.figure()
    f1,f2 = features[0],features[1]
    class0 = np.where(Y == 0)
    class1 = np.where(Y == 1)
    class2 = np.where(Y == 2)
    class3 = np.where(Y == 3)
    if (len(features) == 3):
        #3D scatter plot
        ax = Axes3D(fig)
        f3 = features[2]
        ax.set_zlabel('feature #' + str(f3))
        ax.scatter(X[class0, f1],X[class0, f2], X[class0, f3], marker='+', c='r', label = 'range 0')
        ax.scatter(X[class1, f1], X[class1, f2], X[class1, f3], marker='+', c='g', label = 'range 1')
        ax.scatter(X[class2, f1],X[class2, f2], X[class2, f3], marker='+', c='b', label = 'range 2')
        ax.scatter(X[class3, f1], X[class3, f2], X[class3, f3], marker='+', c='y', label = 'range 3')
    else:
        #2D scatter plot
        plt.scatter(X[class0, f1],X[class0, f2], marker='+', c='r', label = 'range 0')
        plt.scatter(X[class1, f1], X[class1, f2], marker='+', c='g', label = 'range 1')
        plt.scatter(X[class2, f1],X[class2, f2], marker='+', c='b', label = 'range 2')
        plt.scatter(X[class3, f1], X[class3, f2], marker='+', c='y', label = 'range 3')
    plt.legend()
    plt.xlabel('feature #' + str(f1))
    plt.ylabel('feature #' + str(f2))
    return fig

def proportions(Y):
    """
    Gives the proportion of each class in a dataset.
    input: 
      Y : array containing the class of each example in the dataset
    """
    m = np.shape(Y)[0]
    res = np.zeros(4)
    for i in range(4):
        res[i] = np.sum(Y == i)/m
    return res

def learningcurve(Xtrain,ytrain,Xval,yval,fit,predict,error,batch):
    
    mtrain = np.shape(Xtrain)[0]
    
    err = np.zeros(int(mtrain/batch)-1,)
    errval = np.zeros(int(mtrain/batch)-1,)
    
    for i in range(1,int(mtrain/batch)):
        Xi = Xtrain[0:i*batch]
        yi = ytrain[0:i*batch]
        
        fit(Xi,yi)
        pred = predict(Xi)
        err[i-1] = error(pred,yi)
            
        pred = predict(Xval)
        errval[i-1] = error(pred,yval)
        
    
    # Display the learning curves
    fig = plt.figure()
    plt.plot(err, label = "Train")
    plt.plot(errval, label = "Validation")
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    return fig

#%% paramter tunning

def kfolds(X,y,fit,predict,error,k):
    """
    Computes the precision given by error using the k-folds crossvalidation on X
    and using the function fit to train and predict to do predictions
    """
    kf = KFold(n_splits = k)
    err = 0
    errval = 0
    for train_index, val_index in kf.split(X):                             
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        fit(X_train,y_train)
        
        pred = predict(X_train)
        err = err + error(pred,y_train)
        
        pred = predict(X_val)
        errval = errval + error(pred,y_val)
        
    return err/k,errval/k

#%% precision measures
 
def accuracy(prediction,ytest):  
    """
    Computes the usual accuracy of a multiclass prediction by given the 
    predictions it makes on a set
    and the true values on that set
    """
    m = np.shape(ytest)[0]
    return np.sum(prediction == ytest)/m

def f1score_multi(prediction,ytest,K):
    """
    Computes the f1 score of a multiclass prediction by doing the average of 
    the f1 scores given the predictions (as binary labels) it makes on a set
    and the true values on that set
    """
    m = np.shape(ytest)[0]
    f1sum = 0
    for k in range(K):
        
        labels = np.zeros((m,1))
        labels[ytest == k] = 1  # the class at index 0 has value 1
        labels = labels.ravel()
        
        pred = np.zeros((m,1))
        pred[prediction == k] = 1
        pred = pred.ravel()
        
        truepos = np.sum(labels[pred == 1])
        predpos = np.sum(pred)
        actpos = np.sum(labels)
        if predpos != 0 and actpos != 0:
            precision = truepos/predpos
            recall = truepos/actpos
            if precision != 0 and recall != 0:
                f1sum = f1sum + 2*precision*recall/(precision + recall)
        
    return f1sum/K

        