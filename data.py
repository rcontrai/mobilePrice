"""
data.py :
provides tools for loading, preprocessing, visualizing the data
"""

from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
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

def normalizar(X):
    """
    Normalizes the dataset vector X
    and returns the normalized dataset
    along with the means and the standard deviantion
    for each feature
    """
    mu = np.mean(X,  axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

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


def polynomial(X,grad):
    """
    Computes each power of X's values from 0 to grad
    and then normalizes them.
    """
    poly = PolynomialFeatures(grad)
    X_pol = poly.fit_transform(X)
    X_pol[:,1:], mu, sigma = normalizar(X_pol[:,1:])
    return X_pol, mu, sigma
    

#%% visualization

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

