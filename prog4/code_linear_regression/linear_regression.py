########## >>>>>> Put your full name and 6-digit EWU ID here. 


# Implementation of the linear regression with L2 regularization.
# It supports the closed-form method and the gradient-desecent based method. 



import numpy as np
import math
import sys
sys.path.append("..")

from misc.utils import MyUtils


class LinearRegression:
    def __init__(self):
        self.w = None   # The (d+1) x 1 numpy array weight matrix
        self.degree = 1
        
        
    def fit(self, X, y, CF = True, lam = 0, eta = 0.01, epochs = 1000, degree = 1):
        ''' Find the fitting weight vector and save it in self.w. 
            
            parameters: 
                X: n x d matrix of samples, n samples, each has d features, excluding the bias feature
                y: n x 1 matrix of lables
                CF: True - use the closed-form method. False - use the gradient descent based method
                lam: the ridge regression parameter for regularization
                eta: the learning rate used in gradient descent
                epochs: the maximum epochs used in gradient descent
                degree: the degree of the Z-space
        '''
        self.degree = degree
        X = MyUtils.z_transform(X, degree = self.degree)
        
        if CF:
            self._fit_cf(X, y, lam)
        else: 
            self._fit_gd(X, y, lam, eta, epochs)
 


            
    def _fit_cf(self, X, y, lam = 0):
        ''' Compute the weight vector using the clsoed-form method.
            Save the result in self.w
        
            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''

        ## Delete the `pass` statement below.
        ## Enter your code here that implements the closed-form method for
        ## linear regression 

        n,d = X.shape
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        self.w = np.zeros(d+1).reshape((d+1,1))
        
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        return self.w
    
    def _fit_gd(self, X, y, lam = 0, eta = 0.01, epochs = 1000):
        ''' Compute the weight vector using the gradient desecent based method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''

        ## Delete the `pass` statement below.
        
        ## Enter your code here that implements the gradient descent based method
        ## for linear regression 
        n,d = X.shape
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        self.w = np.zeros(d+1).reshape((d+1,1))
        
        p1 = np.identity(d+1) - (((2 * eta) / n) * X.T @ X)
        p2 = ((2 * eta)/n) * X.T @ y
        
        for i in range(epochs):
            self.w = p1 @ self.w + p2
            

        return self.w
    
    def predict(self, X):
        ''' parameter:
                X: n x d matrix, the n samples, each has d features, excluding the bias feature
            return:
                n x 1 matrix, each matrix element is the regression value of each sample
        '''

        ## Delete the `pass` statement below.
        
        ## Enter your code here that produces the label vector for the given samples saved
        ## in the matrix X. Make sure your predication is calculated at the same Z
        ## space where you trained your model. Make sure X has been normalized via the same
        ## normalization used by the training process. 

        n,d = X.shape
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

        # may be wrong computation of predict here
        return X @ self.w
        
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return: 
                the MSE for this test set (X,y) using the trained model
        '''

        ## Delete the `pass` statement below.
        
        ## Enter your code here that calculates the MSE between your predicted
        ## label vector and the given label vector y, for the sample set saved in matraix X
        ## Make sure your predication is calculated at the same Z space where you trained your model. 
        ## Make sure X has been normalized via the same normalization used by the training process. 

        n,d = X.shape
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

        b = X @ self.w
        c = b - y
        d = np.linalg.norm(X @ self.w - y)

        mse = (1/n) * (np.linalg.norm(X @ self.w - y))**2
        
        return mse
 

