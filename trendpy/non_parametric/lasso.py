# -*- coding: utf-8 -*-

# lasso.py

from __future__ import absolute_import

from numpy import eye, zeros, dot, array, diag, sqrt, mean

from scipy.stats import multivariate_normal, invgamma, invgauss, gamma
from numpy.linalg import inv, norm

from trendpy.globals import derivative_matrix

class L1Filter(Sampler):

	def __init__(self,data,alpha=0.1,rho=0.1,total_variation_order=2):
		self.rho = rho
		self.alpha = alpha
		self.__data = data
		self.size = len(data)
		self.total_variation_order = total_variation_order
		self.parameters = None
		self.derivative_matrix = derivative_matrix(self.size, self.total_variation_order)

	@property
	def data(self):
		return self.__data
		
	@property
	def parameters(self):
		""" List containing the parameters to estimate."""
		return self.__parameters

	@parameters.setter
	def parameters(self, new_value):
		self.__parameters = new_value if new_value is not None else Parameters()

	def define_parameters(self):
		params=Parameters()

		params.append(Parameter("trend", multivariate_normal, (self.size,1)))
		params.append(Parameter("sigma2", invgamma, (1,1)))
		params.append(Parameter("lambda2", gamma, (1,1)))
		params.append(Parameter("omega", invgauss, (self.size-self.total_variation_order,1)))

		self.parameters = params

	def initial_value(self,parameter_name):
		if parameter_name=='trend':
			return array([(4*i+10)/20 for i in range(self.size)])
		elif parameter_name=='sigma2':
			return 0.8
		elif parameter_name=='lambda2':
			return 1
		elif parameter_name==str('omega'):
			return 0.8*array([(30*(i/2)+3)/(2*(i/2)+35) for i in range(self.size-self.total_variation_order)])

	def distribution_parameters(self, parameter_name):
		if parameter_name=='trend':
			E = dot(dot(self.derivative_matrix.T,inv(diag(self.parameters.list['omega'].current_value))),self.derivative_matrix)
			mean = dot(inv(eye(self.size)+E),self.data)
			cov = (self.parameters.list['sigma2'].current_value)*inv(eye(self.size)+E)
			return {'mean' : mean, 'cov' : cov}
		elif parameter_name=='sigma2':
			E = dot(dot(self.derivative_matrix.T,inv(diag(self.parameters.list['omega'].current_value))),self.derivative_matrix)
			pos = self.size
			loc = 0
			scale = 0.5*dot((self.data-dot(eye(self.size),self.parameters.list['trend'].current_value)).T,(self.data-dot(eye(self.size),self.parameters.list['trend'].current_value)))+0.5*dot(dot(self.parameters.list['trend'].current_value.T,E),self.parameters.list['trend'].current_value)
		elif parameter_name=='lambda2':
			pos = self.size-self.total_variation_order-1+self.alpha
			loc = 0.5*(norm(dot(self.derivative_matrix,self.parameters.list['trend'].current_value),ord=1))/self.parameters.list['sigma2'].current_value+self.rho
			scale = 1
		elif parameter_name==str('omega'):
			pos = [sqrt(((self.parameters.list['lambda2'].current_value**2)*self.parameters.list['sigma2'].current_value)/(dj**2)) for dj in dot(self.derivative_matrix,self.parameters.list['trend'].current_value)]
			loc = 0
			scale = self.parameters.list['lambda2'].current_value**2
		return {'pos' : pos, 'loc' : loc, 'scale' : scale}

	def generate(self,parameter_name):
		distribution = self.parameters.list[parameter_name].distribution
		parameters = self.distribution_parameters(parameter_name)

		if parameter_name=='trend':
			return distribution.rvs(parameters['mean'],parameters['cov'])
		elif parameter_name=='omega':
			return array([1/distribution.rvs(parameters['pos'][i],loc=parameters['loc'],scale=parameters['scale']) for i in range(len(self.parameters.list['omega'].current_value))]).reshape(self.parameters.list['omega'].current_value.shape)
		return distribution.rvs(parameters['pos'],loc=parameters['loc'],scale=parameters['scale']) #pb with the parameter name

	def output(self, simulations, burn, parameter_name):
		out = mean(simulations[parameter_name][:,:,burn:],axis=2)
		return out

	class Factory(object):
		def create(self,*args,**kwargs):
			return Lasso(args[0],total_variation_order=kwargs['total_variation_order'])
