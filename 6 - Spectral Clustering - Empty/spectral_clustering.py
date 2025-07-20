import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.ion()


def spectral_clustering(data, n_cl, sigma=1., fiedler_solution=False):
	return ...


def main_spectral_clustering():
	"""
	Main function for spectral clustering.
	"""

	# generate the dataset
	data, cl = two_moon_dataset(n_samples=300, noise=0.1)  # best sigma = 0.1
	# data, cl = gaussians_dataset(n_gaussian=3, n_points=[100, 100, 70], mus=[[1, 1],
	# 								[-4, 6], [8, 8]], stds=[[1, 1], [3, 3], [1, 1]])  # best sigma = 2

	# visualize the dataset
	_, ax = plt.subplots(1, 2)
	ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)

	# run spectral clustering
	labels = spectral_clustering(data, n_cl=2, sigma=0.1, fiedler_solution=True) # two moons
	#labels = spectral_clustering(data, n_cl=3, sigma=2, fiedler_solution=False) # gaussians

	# visualize results
	ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
	plt.waitforbuttonpress()


if __name__ == '__main__':
	main_spectral_clustering()
