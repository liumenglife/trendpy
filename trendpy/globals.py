# -*- coding: utf-8 -*-

# globals.py

from __future__ import absolute_import

from numpy import zeros

__all__ = ['derivative_matrix']

DATE_FORMAT = '%m/%d/%Y'

DATETIME_FORMAT = '%a, %b %d %Y %H:%M:%S'

def derivative_matrix(size, order=2):
	""" Computes a discrete difference operator.

	:param size: dimension of the matrix.
	:type size: int
	:param order: derivation order.
	:type order: int
	:return: Discrete difference operator
	:rtype: `Numpy.dnarray`
	"""
	D=zeros((size-order, size))
	if order==0:
		d=[1]
	elif order==1:
		d=[-1,1]
	elif order==2:
		d=[1,-2,1]
	elif order==3:
		d=[-1,3,-3,1]
	for n in range(size-order):
		for l in range(order+1):
			D[n,n+l]=d[l]
	return D
