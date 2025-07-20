import numpy as np


eps = np.finfo(float).eps


def sigmoid(x):
	return ...


def loss(y_true, y_pred):
	return ...


def dloss_dw(y_true, y_pred, X):
	return ...


class LogisticRegression:
	""" Models a logistic regression classifier. """

	def __init__(self):
		""" Constructor method """

		# weights placeholder
		self._w = None

	def fit_gd(self, X, Y, n_epochs, learning_rate, verbose=False):
		pass


	def predict(self, X):
		return ...
