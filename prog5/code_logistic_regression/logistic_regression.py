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
        self.degree = degree
        
        # X --> Z(X) --> add bias column --> X
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        n,d = X.shape

        #init self.w same as you did for linear regression. The self.w size is d x 1
        self.w = np.zeros(d).reshape((d,1))

        # permute X and y in unison
        X, y = LogisticRegression._unison_shuffle(X, y)
        
        if SGD is False: 
            for i in range(iterations): 
                s = y * (X @ self.w)
                self.w = (1.0 - 2.0*lam*eta/n) * self.w + eta/n * (X.T @ (y * LogisticRegression._v_sigmoid(s * (-1.0)) ))
        
        else: # SGD is True

            import itertools

            # get indices of the mini batches, 
            # has form mini_batch_indices[i] = [starting index, ending index]
            mini_batch_indices = LogisticRegression._mini_batch_indices(n, mini_batch_size)

            # create iterator for cycling through mini_batch_indices array
            cycle = itertools.cycle(mini_batch_indices)

            for i in range(iterations): 

                # each iteration of for loop needs to calculate new indices for X' and y'
                # w/out creating copies of X, y 
                # need to continually cycle through mini_batch_indices array to get desired indices for current self.w update
                repeat_indices = next(cycle)
                n_prime = repeat_indices[1] - repeat_indices[0] + 1
                s = y[repeat_indices[0]: (repeat_indices[1] + 1)] * (X[repeat_indices[0]: repeat_indices[1], :] @ self.w)
                self.w = (1.0 - 2.0*lam*eta/n_prime) * self.w + eta/n_prime * (X[repeat_indices[0]:  (repeat_indices[1] + 1), :].T @ (y[repeat_indices[0]: repeat_indices[1]] * LogisticRegression._v_sigmoid(s * (-1.0))))    

    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
    
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
        vs = np.vectorize(LogisticRegression._sigmoid, otypes=[float])
        return vs(s)
    
        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''
    
        return 1.0 / (1.0 + math.exp(-s))
    
    def _unison_shuffle(a, b):
        
        # function to perform unison shuffling using random permutation
        if(len(a) == len(b)):
            perm = np.random.permutation(len(a))
            return a[perm], b[perm]
        else: 
            raise Exception("Array lengths must match to perform unison shuffle.")
        
    
    def _mini_batch_indices(n, mini_batch_size):

        # function to compute mini batch indices
        # returns array of mini batch indices in the form
        # index_arr[i] = [starting index, ending index]
        remainder_minibatch_length = n % mini_batch_size 
        index_arr = []

        for i in range(0, n - remainder_minibatch_length, mini_batch_size):
            temp = np.arange(2)
            temp[0], temp[1] = i, ((i + mini_batch_size) - 1)
            index_arr.append(temp)
            
        if(remainder_minibatch_length != 0):    
            temp_remainder = np.arange(2)
            temp_remainder[0], temp_remainder[1] = (n - remainder_minibatch_length), (n - 1)
            index_arr.append(temp_remainder)

        return index_arr
        
        
