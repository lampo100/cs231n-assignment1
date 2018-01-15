import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax(X):
  """
  Softmax function.
  
  Inputs:
  - X: A numpy array of shape (N, ).
  
  Returns a numpy array of shape (N,) containing softmax values of elements of array X.
  """
  X = X - np.max(X)
  e_X = np.exp(X)
  return (e_X/np.sum(e_X, axis=1)[:, np.newaxis])
    

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
    

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train, num_features = X.shape
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  predicted_scores = X.dot(W)
  softmax_scores = softmax(predicted_scores)
  for i in range(num_train):
    Li = -1 * np.log(softmax_scores[i, y[i]])
    loss += Li
  
  for i in range(num_train):
    for j in range(num_class):
        if(j == y[i]):
            dW[:, j] += -X[i, :] * (1 - softmax_scores[i, j])
        else:
            dW[:, j] += X[i, :] * softmax_scores[i, j]

    
  dW /= num_train
  loss /= num_train

  regularization_cost = np.sum(np.square(W))
  loss += reg*regularization_cost
  dW += 2 * reg * W 
    

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train, num_features = X.shape
  num_class = W.shape[1]
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  predicted_scores = X.dot(W)
  softmax_scores = softmax(predicted_scores)
  loss = -1 * np.sum(np.log(softmax_scores[np.arange(num_train), y]))
    
  loss /= num_train  
  loss += reg * np.sum(np.square(W))
  

  indication_matrix = np.zeros(softmax_scores.shape)
  indication_matrix[np.arange(num_train), y] = 1
  indication_matrix -= softmax_scores

  dW = np.transpose(X).dot(indication_matrix)
   
  dW /= -num_train
  dW += 2 * reg * W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

