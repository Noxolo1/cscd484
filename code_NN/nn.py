# Nate Wilson - 00958137

### Delete every `pass` statement below and add in your own code. 



# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



import numpy as np
import math
import math_util as mu
import nn_layer


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        new_layer = nn_layer.NeuralLayer(d, act)
        self.layers.append(new_layer)
        self.L += 1
        
    
    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''

        # need to walk through every layer in self.layers
        # and initialize self.w (the weights) of the given layer using 
        # range above 
        # start range at 1 so we can get d value for l - 1 layer
        for i in range(1, self.L + 1):
            
            d_cur = self.layers[i].d
            sqrt_d_cur = np.sqrt(d_cur)
            d_prev = self.layers[i - 1].d

            # might need to change this, currently excludes high
            vec = np.random.uniform(-1.0/sqrt_d_cur, 1.0/sqrt_d_cur, (d_prev + 1, d_cur))

            # (d^{(\ell-1)}+1 ) x d^{(\ell)} matrix. The weights of the edges coming into layer \ell.
            # repeatedly insert vec as a column into self.w to create (might need to insert as rows
            # but shouldn't matter here)
            # (d^{(\ell-1)}+1 ) x d^{(\ell)} matrix
            self.layers[i].W = vec
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.

        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 

        # prep data: add bias column and shuffle
        X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
        n,d = X.shape
        X, Y = self._unison_shuffle(X, Y)

        import itertools

        mini_batch_indices = self._mini_batch_indices(n, mini_batch_size)

        # create iterator for cycling through mini_batch_indices array
        cycle = itertools.cycle(mini_batch_indices)

        for i in range(1, iterations + 1):

            # get a minibatch
            # returns array of mini batch indices in the form
            # index_arr[i] = [starting index, ending index]
            repeat_indices = next(cycle)
            start = repeat_indices[0]
            end = repeat_indices[1]

            X_prime = X[start: end + 1, :]
            Y_prime = Y[start: end + 1, :]
            n_prime, d_prime = X_prime.shape

            self.layers[0].X = X_prime

            # forward feed with minibatch
            self._forward_feed()

            # calculate error
            E = np.sum((self.layers[self.L].X[:,1:] - Y_prime) * (self.layers[self.L].X[:,1:] - Y_prime)) * (1/n_prime)

            # calculate delta 
            self.layers[self.L].Delta = 2 * (self.layers[self.L].X[:,1:] - Y_prime) * self.layers[self.L].act_de(self.layers[self.L].S)
            
            # calculate gradient
            self.layers[self.L].G = np.einsum('ij, ik -> jk', self.layers[self.L - 1].X, self.layers[self.L].Delta)* (1/n_prime)

            ### back propagrate
            self._back_propagation(n_prime)

            self._update_weights(eta)

    def _forward_feed(self):
        
        for i in range(1, self.L + 1):

            self.layers[i].S = self.layers[i-1].X @ self.layers[i].W
            theta_S_l = self.layers[i].act(self.layers[i].S)
            self.layers[i].X = np.insert(theta_S_l, 0, np.ones(theta_S_l.shape[0]), axis = 1)    
    
    def _back_propagation(self, n_prime):

        for i in range(self.L - 1, 0, -1):
            
            # W might have wrong indices being used here
            self.layers[i].Delta = self.layers[i].act_de(self.layers[i].S) * (self.layers[i + 1].Delta @ (self.layers[i + 1].W[1:, :]).T)
            
            self.layers[i].G = np.einsum('ij, ik -> jk', self.layers[i-1].X, self.layers[i].Delta)* (1/n_prime)

    def _update_weights(self, eta):
        for i in range(1, self.L + 1):
            self.layers[i].W -= (eta * self.layers[i].G)
            
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
            
            Note that the return of this function is NOT the sames as the return of the 
            `NN_Predict` method in the lecture slides. In fact, every element in the 
            vector returned by this function is the column index of the largest 
            number of each row in the matrix returned by the `NN_Predict` method in the lecture slides.
         '''
        self.layers[0].X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

        self._forward_feed()  

        return np.argmax(self.layers[self.L].X[:, 1:], axis=1)    
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        
       	n = X.shape[0]  # Number of samples
        predicted_labels = self.predict(X)  # Predict labels for input samples
        
        # Convert predicted labels and ground truth labels to class indices
        true_labels = np.argmax(Y, axis=1)
        
        # Count misclassified samples
        misclassified_count = np.sum(predicted_labels != true_labels)
        
        # Calculate misclassification rate        
        return misclassified_count / n

    def _unison_shuffle(self, a, b):
        
        # function to perform unison shuffling using random permutation
        if(len(a) == len(b)):
            permutation = np.random.permutation(len(a))
            return a[permutation], b[permutation]
        
    def _mini_batch_indices(self, n, mini_batch_size):

        # function to compute mini batch indices
        # returns array of mini batch indices in the form
        # index_arr[i] = [starting index, ending index]
        remainder_minibatch_length = n % mini_batch_size 
        index_arr = []

        for i in range(0, n - remainder_minibatch_length, mini_batch_size):
            temp = [i, ((i + mini_batch_size) - 1)]
            index_arr.append(temp)
            
        if(remainder_minibatch_length != 0):    
            temp_remainder = [(n - remainder_minibatch_length), (n - 1)]
            index_arr.append(temp_remainder)

        return np.asarray(index_arr)