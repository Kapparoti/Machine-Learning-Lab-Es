import matplotlib.pyplot as plt
import numpy as np

from utils import cmap


class WeakClassifier:
	"""
	Function that models a WeakClassifier
	"""

	def __init__(self):
		# initialize a few stuff
		self._dim = None
		self._threshold = None
		self._label_above_split = None

	def fit(self, X: np.ndarray, Y: np.ndarray):
		pass

	def predict(self, X: np.ndarray):
		return ...


class AdaBoostClassifier:
	"""
	Function that models a Adaboost classifier
	"""

	def __init__(self, n_learners: int, n_max_trials: int = 200):
		"""
		Model constructor

		Parameters
		----------
		n_learners: int
			number of weak classifiers.
		"""

		# initialize a few stuff
		self.n_learners = n_learners
		self.learners = []
		self.alphas = np.zeros(shape=n_learners)
		self.n_max_trials = n_max_trials

	def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
		pass

	def predict(self, X: np.ndarray):
		return ...

	def _plot(self, X: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
			  learner: WeakClassifier, iteration: int):

		# plot
		plt.clf()
		plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=weights * 50000,
					cmap=cmap, edgecolors='k')

		M1, m1 = np.max(X[:, 1]), np.min(X[:, 1])
		M0, m0 = np.max(X[:, 0]), np.min(X[:, 0])

		cur_split = learner._threshold
		if learner._dim == 0:
			plt.plot([cur_split, cur_split], [m1, M1], 'k-', lw=5)
		else:
			plt.plot([m0, M0], [cur_split, cur_split], 'k-', lw=5)
		plt.xlim([m0, M0])
		plt.ylim([m1, M1])
		plt.xticks([])
		plt.yticks([])
		plt.title('Iteration: {:04d}'.format(iteration))
		plt.waitforbuttonpress(timeout=0.1)
