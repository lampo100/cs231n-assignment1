import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        scalar_value = sum((scores - correct_class_score + 1) > 0) - 1
        dW[:, j] += -1 * scalar_value * X[i]
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]      
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #Compute our scores
  predicted_scores = np.dot(X, W)
  #Make an array of scores of the correct class in each row
  correct_classes_scores = predicted_scores[np.arange(y.size), y]
  #Compute max(s_j - s_y + 1) in each cell
  predicted_scores -= correct_classes_scores[:,np.newaxis]
  predicted_scores += 1
  predicted_scores[predicted_scores<0] = 0
  
  #Check which cells are greater than 0 and set correct classes cells to 0
  bool_of_margins = predicted_scores
  bool_of_margins[predicted_scores > 0] = 1
  bool_of_margins[np.arange(y.size), y] = 0

  #Sum every cell for our loss function
  loss = np.sum(predicted_scores[bool_of_margins > 0])
  
  #For gradient get loss count(classes that have score higher that our true class \
  #for each example 
  vert_sum = np.sum(bool_of_margins, axis = 1)
  #Put first counted loss in the first row true class place, second counted loss in second loss etc.
  bool_of_margins[np.arange(y.size), y] = -vert_sum.T
  #Multiply to get our gradient Matrix
  dW = np.dot(X.T, bool_of_margins)
  
  #Average and add normalization factor  
  dW /= y.size
  
  dW += 2 * reg * W #Regularize
  loss /= y.size #### <<<<<<<<<<<<<<<<<<<< WE WANT AVERAGE!!!!!!!!!!
  squared_W = np.square(W)
  loss += np.sum(reg * squared_W)  #Regularize  


  return loss, dW
