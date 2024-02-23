########## >>>>>> Nate Wilson - 00958137. 

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent




import numpy as np
import math
import sys
sys.path.append("..")

from code_misc.utils import MyUtils


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1

        

    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''

        # remove the pass statement and fill in the code. 

        #pass
        self.degree = degree
        
        # X --> Z(X) --> add bias column --> X
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        n,d = X.shape
        self.w = np.zeros(d).reshape((d,1))

        # permute X and y in unison
        X, y = LogisticRegression._unison_shuffle(X, y)
        
        #init self.w same as you did for linear regression. The self.w size is d x 1
        
        if SGD is False: 
            for i in range(iterations): 
                s = y * (X @ self.w)
                self.w = (1.0 - 2.0*lam*eta/n) * self.w + eta/n * (X.T @ (y * LogisticRegression._v_sigmoid(s * (-1.0)) ))
        
        else: # SGD is True

            for i in range(0, iterations, mini_batch_size): 

                X_prime, y_prime = X[i: (i + mini_batch_size), 0: d], y[i: i + mini_batch_size]

                s = y_prime * (X_prime @ self.w)
                self.w = (1.0 - 2.0*lam*eta/mini_batch_size) * self.w + eta/mini_batch_size * (X_prime.T @ (y_prime * LogisticRegression._v_sigmoid(s * (-1.0))))    

    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
    
        # remove the pass statement and fill in the code. 
        #pass
        # X -> Z(X) -> add bias column --> X
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        
        return LogisticRegression._v_sigmoid(X @ self.w)
       
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''

        # remove the pass statement and fill in the code.         
        #pass
        # X -> Z(X) -> add bias column --> X
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

        return np.sum(np.sign((np.sign(X @ self.w) - 0.1)) != y)
      
    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
            
        # Hint: use the np.vectorize API

        # remove the pass statement and fill in the code.         

        vs = np.vectorize(LogisticRegression._sigmoid, otypes=[float])
        b = vs(s)
        return b
    
        #return np.vectorize(LogisticRegression._sigmoid)(s)
    
        #pass
    
    
        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''

        # remove the pass statement and fill in the code.         
        
        return 1.0 / (1.0 + math.exp(-s))
    
        #pass
    
    def _unison_shuffle(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]