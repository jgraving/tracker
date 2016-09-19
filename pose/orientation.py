#! /usr/bin/env python

from __future__ import division, print_function
import numpy as np


def get_orientation(contour, C = 0.3, degrees = True):

	"""Returns the orientation information of a contour

		Parameters
		----------
		contour : numpy_array
			contour array from OpenCV
		C : int or float, default = 0.3
			constant to multiply the eigenvalues for the axes coordinates
		degrees : bool, default = True
			return orientation in degrees, not radians

		Returns
		-------
		axes : dict
			dict of axes coordinates, angles, and eigenvalue ratio
			"major" : numpy_array, major axis coordinates (centroid, C * eigenvalue) 
			"minor" : numpy_array, minor axis coordinates (centroid, C * eigenvalue)
			"o1" : float, major axis orientation 
			"o2" : float, minor axis orientation
			"ratio" : float, ratio of major/minor axes eigenvalues (measure of circularity)
		"""

	# reshape the contour to something sensible
	contour = contour.reshape(contour.shape[0],2)
	x = contour[:,0]
	y = contour[:,1]

	# subtract the mean from each variable (column) in the data
	mx = x - np.mean(x)
	my = y - np.mean(y)
	data = np.array([mx,my])

	# decompose the contour with SVD
	eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)

	# get coordinates for decomposed axes
	PC1_eigen = C * eigenvalues[0]
	PC2_eigen = C * eigenvalues[1]
	PC1_x = [0,V[0,0]*PC1_eigen]+np.mean(x)
	PC1_y = [0,V[0,1]*PC1_eigen]+np.mean(y)
	PC2_x = [0,V[1,0]*PC2_eigen]+np.mean(x)
	PC2_y = [0,V[1,1]*PC2_eigen]+np.mean(y)
	
	major = np.array([PC1_x,PC1_y]).T
	minor = np.array([PC2_x,PC2_y]).T
	
	# get eigenvalue ratio
	ratio = eigenvalues[0]/eigenvalues[1]

	# get orientation of the axes
	PC1_xdiff = PC1_x[1] - PC1_x[0]
	PC1_ydiff = PC1_y[1] - PC1_y[0]
	PC2_xdiff = PC2_x[1] - PC2_x[0]
	PC2_ydiff = PC2_y[1] - PC2_y[0]

	orientation1 = np.arctan2(PC1_ydiff,PC1_xdiff)
	orientation2 = np.arctan2(PC2_ydiff,PC2_xdiff)

	# convert radians to degrees if specified
	if degrees == True:
		orientation1 = np.degrees(orientation1)
		orientation2 = np.degrees(orientation2)

	# make dict	
	axes = {"major" : major, "minor" : minor, "o1" : orientation1, "o2" : orientation2, "ratio" : ratio}
	
	return axes