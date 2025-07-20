from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform


class KMeans:

	def __init__(self, n_cl: int, n_init: int = 1,
				 initial_centers: Optional[np.ndarray] = None,
				 verbose: bool = False) -> None:
		"""
		Parameters
		----------
		n_cl: int
			number of clusters.
		n_init: int
			number of time the k-means algorithm will be run.
		initial_centers:
			If an ndarray is passed, it should be of shape (n_clusters, n_features)
			and gives the initial centers.
		verbose: bool
			whether or not to plot assignment at each iteration (default is True).
		"""

		self.n_cl = n_cl
		self.n_init = n_init
		self.initial_centers = initial_centers
		self.verbose = verbose

	def _init_centers(self, X: np.ndarray, use_samples: bool = False):
		return ...

	def single_fit_predict(self, X: np.ndarray):
		return ...

	def compute_cost_function(self, X: np.ndarray, centers: np.ndarray, assignments: np.ndarray):
		return ...

	def fit_predict(self, X: np.ndarray):
		return ...
