import numpy as np
import matplotlib.pyplot as plt
import random

class RegularizedLogisticRegression(object):
	'''Implements regularized logistic regression for binary classification.

	The weight vector w should be learned by minimizing the regularized risk
	log(1 + exp(-y <w, x>)) + \lambda \|w\|_2^2. In other words, the objective
	function is the log loss for binary logistic regression plus Tikhonov
	regularization with a coefficient of \lambda.
	'''

	'''Implements regularized logistic regression for binary classification.

	The weight vector w should be learned by minimizing the regularized risk
	-1 * \frac{1}{m} \sum_{i}^{m} (y_{i} * log(h(x_{i}))) + (1 - y_{i}) * log(1 - h(x_{i})))
    + \lambda \|w\|_2^2, where the h(x) is the sigmoid function in this
    case. In other words, the objective function is the log loss for binary
    logistic regression plus Tikhonov regularization with a coefficient of \lambda.
	'''

	def __init__(self):
		self.learningRate = 0.00001 # Please dont change this
		self.num_epochs = 100000 # Feel free to play around with this if you'd like, though this value will do

		#####################################################################
		#																	#
		#	MAKE SURE TO SET THIS TO THE OPTIMAL LAMBDA BEFORE SUMITTING	#
		#																	#
		#####################################################################

		self.lmbda = 10 # tune this parameter

	def train(self, X, Y):
		'''
        Trains the model, using stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None
        '''

        # initialize weights
		self.weights = np.zeros(len(np.transpose(X)))
		# run for number of epochs we are given
		for i in range(self.num_epochs):
			# calculate gradient
			l = np.dot(X,self.weights)
			p = sigmoid_function(l)
			loss_function = (1/len(X)) * np.dot((p - Y),X)
			regularization_term = self.lmbda * 2 * self.weights
			gradientlw = loss_function + regularization_term
			# find new weights using gradient
			wnew = self.weights - (self.learningRate * gradientlw)
			# update weights
			self.weights = wnew



	def predict(self, X):
		'''
        Compute predictions based on the learned parameters and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
		labels = np.zeros(len(X))
		predictions = np.dot(X,self.weights)
		probabilities = sigmoid_function(predictions)
		labels = np.where(probabilities >= 0.5, labels + 1, labels)
		return(labels)


	def accuracy(self,X,Y):
		'''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
		return(np.mean(self.predict(X) == Y))


	def plotError(self, X_train, y_train, X_val, y_val):
		'''
		Produces a plot of the cost function on the training and validation
		sets with respect to the regularization parameter lambda. Use this function to determine
		a valid lambda
		@params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
		'''
		lambda_list = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]
		#self.lmbda = lambda_list
		error_list_train = []
		error_list_val = []
		#[TODO] train model and calculate train and validation errors here for each lambda, then plot.
		for i in range(len(lambda_list)):
			self.lmbda = lambda_list[i]
			self.train(X_train, y_train)
			error_list_train.append(1 - self.accuracy(X_train,y_train))
			error_list_val.append(1 - self.accuracy(X_val,y_val))
			#print(error_list_train)
			#print(error_list_val)

		'''
		y1 = np.array(error_list_train)
		y2 = np.array(error_list_val)
		x = np.array(lambda_list)
		plt.xscale("log")
		plt.xlim((100000,0.0001))
		plt.scatter(x, y1, c='r', label="Training")
		plt.scatter(x, y2, c='b', label="Validation")
		plt.title('Made by: d85eba7e', fontsize=16)
		plt.legend()
		plt.show()
		'''

def sigmoid_function(x):
	return 1.0 / (1.0 + np.exp(-x))
