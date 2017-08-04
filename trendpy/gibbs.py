# -*- coding: utf-8 -*-

# gibbs.py

from __future__ import absolute_import

from numpy import (reshape, zeros)

class Gibbs(object):

	def __init__(self, sampler):
		self.sampler = sampler
		self.simulations = None

	def define_parameters(self):
		""" Method to set the parameter set to be updated
			in the MCMC algorithm.
		"""
		return self.sampler.define_parameters()

	def initial_value(self,parameter_name):
		""" Method that sets the initial value of the
			parameters to be estimated.

		:param parameter_name: name of the parameter.
		:type parameter_name: str
		:return: initial value of the parameter
		:rtype: `Numpy.dnarray`
        """
		return self.sampler.initial_value(parameter_name)

	def distribution_parameters(self, parameter_name):
		""" Method that sets the parameters of the posterior
			distribution of the parameters to be estimated.

		:param parameter_name: name of the parameter.
		:type parameter_name: str
		:return: dictionary the parameters needed to compute the
			next value of the Markov chain for the parameter with name:
			parameter_name.
		:rtype: dict
        """
		return self.sampler.distribution_parameters(parameter_name) # returns a dictionary

	def generate(self, parameter_name):
		""" This method handles the generation of the random draws of
			the Markov chain for each parameters.

		:param parameter_name: name of the parameter of interest
		:type parameter_name: string
		:return: random draw from the posterior probability distribution
		:rtype: `Numpy.dnarray`
        """
		return self.sampler.generate(parameter_name)

	def output(self, burn, parameter_name):
		""" Computes the poserior mean of the parameters.

		:param parameter_name: name of the parameter of interest
		:type parameter_name: string
		:param burn: number of draws dismissed as burning samples
		:type burn: int
		:return: output of the MCMC algorithm
		:rtype: `Numpy.dnarray`
        """
		return self.sampler.output(self.simulations, burn, parameter_name)

	def run(self, number_simulations=100, max_restart=10):
		""" Runs the MCMC algorithm.

		:param number_simulations: number of random draws for each parameter.
		:type number_simulations: int
		"""
		self.simulations = {key : zeros((param.size[0],param.size[1],number_simulations)) for (key, param) in self.sampler.parameters.list.items()}

		for name in self.sampler.parameters.hierarchy:
			self.sampler.parameters.list[name].current_value = self.initial_value(name)

		for i in range(number_simulations):
			print("== step %i ==" % (int(i+1),))
			restart = 0
			restart_step = True
			while restart_step:
				for name in self.sampler.parameters.hierarchy:
					print("== parameter %s ==" % name)
					try:
						self.sampler.parameters.list[name].current_value = self.generate(name)
						self.simulations[name][:,:,i] = self.sampler.parameters.list[name].current_value.reshape(self.sampler.parameters.list[name].size)
						restart_step = False
						restart = 0
					except:
						if restart < max_restart:
							restart+=1
							print("== restart step %i ==" % i)
							restart_step = True
							break
						else:
							raise ValueError("Convergence error")
