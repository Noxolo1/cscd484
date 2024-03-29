# Implementation of the perceptron learning algorithm. Support the pocket version for linearly unseparatable data. 
# Authro: Bojian Xu, bojianxu@ewu.edu

#Important observation: 
#    - The PLA can increase or decrease $w[0]$ by 1 per update, so if there is a big difference between $w^*[0]$ and the #initial value of $w[0]$, the PLA is likely to take a long time before it halts. However, the theoretical bound $O((L/d)^2)$ #step of course still holds, where $L = \max\{\lVert x\rVert\}$ and $d$ is the margine size.
#    - This can solved by always have feature values within [0,1], because by doing so, the $x_0=1$ becomes relatively larger (or one can also say $x_0$ becomes fairly as important as other feathers), which makes the changes to $w[0]$ much faster. This is partially why nueral network requires all feature value to be [0,1] --- the so-called data normalization process!!!

# Another reason for normalizing the feature into [0,1] is: no matter which Z space the samples are tranformed to, the Z-space sample features will still be in the [0,1] range. 

import numpy as np

#import sys
#sys.path.append("..")

from utils import MyUtils



class PLA:
    def __init__(self, degree=1):
        self.w = None
        self.degree = degree
        
    def fit(self, X, y, pocket = True, epochs = 100):
        ''' find the classifer weight vector and save it in self.w
            X: n x d matrix, i.e., the bias feature is not included. 
            It is assumed that X is already normalized be data preprocessing. 
            y: n x 1 vector of {+1, -1}
            degree: the degree of the Z space
            return self.w
        '''
        
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree=self.degree)

            
        ### BEGIN YOUR SOLUTION
        #raise NotImplementedError()
            
        n,d = X.shape
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        self.w = np.zeros(d+1)
        
        if not pocket:

            update = True
            
            while update:

                update = False

                for i in range(n):
                    if np.sign(X[i,:] @ self.w) != y[i]:

                        self.w += y[i] * X[i,:] 
                        update = True
        else:

            update = True
            best_w = self.w

            while update and epochs > 0:
        
                epochs -= 1
                update = False

                for i in range(n):
                    if np.sign(X[i,:] @ self.w) != y[i]:

                        self.w = self.w + y[i] * X[i,:] 
                        update = True
                        
                        y_hat = np.sign(X @ self.w).reshape((y.shape[0], 1))

                        # calculate misclassified counts
                        error_count = np.sum(y_hat != y)

                        y_hat_best_w = np.sign(X @ best_w).reshape((y.shape[0], 1))

                        # update best_w if the current w is better
                        if error_count < np.sum(y_hat_best_w != y):
                            best_w = self.w

            self.w = best_w

        ### END YOUR SOLUTION
            
                          
        return self.w
    
                          


    def predict(self, X):
        ''' x: n x d matrix, i.e., the bias feature is not included.
            return: n x 1 vector, the labels of samples in X
        '''
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)

    
        ### BEGIN YOUR SOLUTION
        #raise NotImplementedError()
            
        # matrix multiplication Xw, then take sign of result to produce y prediction
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

        return np.sign(X @ self.w)
        ### END YOUR SOLUTION

        
        


    def error(self, X, y):
        ''' X: n x d matrix, i.e., the bias feature is not included. 
            y: n x 1 vector
            return the number of misclassifed elements in X using self.w
        '''
        
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)

        ### BEGIN YOUR SOLUTION
        #raise NotImplementedError()
            
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
                
        y_hat = np.sign(X @ self.w).reshape((y.shape[0], 1))

        # calculate misclassified counts
        error_count = np.sum(y_hat != y)

        return error_count

        ### END YOUR SOLUTION
            


    