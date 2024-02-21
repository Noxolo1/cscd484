##### >>>>>> Nate Wilson - 00958137


# Various tools for data manipulation. 



import numpy as np
import math

class MyUtils:

    
    def z_transform(X, degree = 2):
        ''' Transforming traing samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)
        '''
    
    
    ######### place here the code that you have submitted for the previous programming assignment
        if degree == 1:
            return X

        ### BEGIN YOUR SOLUTION
        #raise NotImplementedError()
        n, d = X.shape

        # dont want to use zero element
        B = np.zeros(degree + 1)
        
        # dont want to use zero element
        for i in range(1, degree + 1):
            B[i] = math.comb((i + d - 1), (d - 1))
        
        d_prime = np.sum(B[1: degree + 2])

        Z = np.copy(X)

        #
        l = np.arange(d_prime, dtype=int)

        q  = 0  # total size of all buckets before the prev bucket
        p = d  # total size of all previous buckets
        g = d  # index of the new column in Z that is being computed

        for i in range(2, int(degree + 1)):
            for j in range (int(q), int(p)):
                for k in range(int(l[j]), int(d)):
                    temp = Z[:,j] * X[:,k] 
                    Z = np.append(Z, temp.reshape(-1,1), 1)
                    l[g] = k
                    g += 1

            q = p
            p += B[i]
        
        return Z

    
    ## below are the code that your instructor wrote for feature normalization. You can feel free to use them
    ## but you don't have to, if you want to use your own code or other library functions. 

    def normalize_0_1(X):
        ''' Normalize the value of every feature into the [0,1] range, using formula: x = (x-x_min)/(x_max - x_min)
            1) First shift all feature values to be non-negative by subtracting the min of each column 
               if that min is negative.
            2) Then divide each feature value by the max of the column if that max is not zero. 
            
            X: n x d matrix of samples, excluding the x_0 = 1 feature. X can have negative numbers.
            return: the n x d matrix of samples where each feature value belongs to [0,1]
        '''

        n, d = X.shape
        X_norm = X.astype('float64') # Have a copy of the data in float

        for i in range(d):
            col_min = min(X_norm[:,i])
            col_max = max(X_norm[:,i])
            gap = col_max - col_min
            if gap:
                X_norm[:,i] = (X_norm[:,i] - col_min) / gap
            else:
                X_norm[:,i] = 0 #X_norm[:,i] - X_norm[:,i]
        
        return X_norm

    def normalize_neg1_pos1(X):
        ''' Normalize the value of every feature into the [-1,+1] range. 
            
            X: n x d matrix of samples, excluding the x_0 = 1 feature. X can have negative numbers.
            return: the n x d matrix of samples where each feature value belongs to [-1,1]
        '''

        n, d = X.shape
        X_norm = X.astype('float64') # Have a copy of the data in float

        for i in range(d):
            col_min = min(X_norm[:,i])
            col_max = max(X_norm[:,i])
            col_mid = (col_max + col_min) / 2
            gap = (col_max - col_min) / 2
            if gap:
                X_norm[:,i] = (X_norm[:,i] - col_mid) / gap
            else: 
                X_norm[:,i] = 0 #X_norm[:,i] - X_norm[:,i]

        return X_norm
