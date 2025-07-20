import numpy as np
from numpy.linalg import norm

eps = np.finfo(float).eps


class SVM:
	""" Models a Support Vector machine classifier based on the PEGASOS algorithm. """

	def __init__(self, n_epochs, lambDa, use_bias=True):
		""" Constructor method """

		# weights placeholder
		self._w = None
		self._original_labels = None
		self._n_epochs = n_epochs
		self._lambda = lambDa
		self._use_bias = use_bias

	def map_y_to_minus_one_plus_one(self, y):
		"""
		Map binary class labels y to -1 and 1
		"""
		ynew = np.array(y)
		self._original_labels = np.unique(ynew)
		assert len(self._original_labels) == 2
		ynew[ynew == self._original_labels[0]] = -1.0
		ynew[ynew == self._original_labels[1]] = 1.0
		return ynew

	def map_y_to_original_values(self, y):
		"""
		Map binary class labels, in terms of -1 and 1, to the original label set.
		"""
		ynew = np.array(y)
		ynew[ynew == -1.0] = self._original_labels[0]
		ynew[ynew == 1.0] = self._original_labels[1]
		return ynew

	def loss(self, y_true, y_pred):
		return ...

	def fit_gd(self, X, Y, verbose=False):
		for e in range(1, self._n_epochs + 1):

			if verbose:
				print("Epoch {} Loss {}".format(e, cur_loss))

	def predict(self, X):
		return ...
