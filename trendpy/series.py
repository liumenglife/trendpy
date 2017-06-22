# -*- coding: utf-8 -*-

# series.py

# MIT License

# Copyright (c) 2017 Rene Jean Corneille

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt


from trendpy.mcmc import MCMC
from trendpy.factory import StrategyFactory

from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params

from datetime import datetime

from pandas import DataFrame, read_csv

class Series(object):
	""" Implements univariate time series.

	Examples
	--------

		Import the class

		>>> from trendpy.series import Series

		create from csv

		>>> data = Series.from_csv('data.csv')
	"""
	def __init__(self):
		self.data=None
		self.begin_index=None
		self.ed_index=None

	def __len__(self):
		return self.data.size

	def __str__(self):
		return self.data.__str__()

	@staticmethod
	def from_csv(filename,index='date'):
		""" Instantiate new time series from a csv file where the first
			column is a timestamp or a date or a datetime.

		:param filename: path of the file with extension (.csv or .txt)
		:type filename: str
		:return: the price time series
		:rtype: `trendpy.series.Series`
		"""
		time_series=Series()
		time_series.data=read_csv(filename,index_col=0)
		return time_series

	def returns(self,period=1,annualize='Y'):
		""" Adds a new time series to the data with the returns of the original
			time series.

		:param period: number of days between two consecutive observations used to
				compute the returns.
		:type period: int, optional
		:return: output of the MCMC algorithm
		:rtype: `Numpy.dnarray`
		"""
		return_series = Series()
		return_series.data = self.data.pct_change(periods=period)
		return_series.data.returns.fillna(value=0)
		return return_series
		
	def summary(self):
		""" Returns an ASCII table with basic statistics of the time series loaded.
		
		:return: 
		:rtype: 
		"""
		smry = Summary()
		summary_title = 'Summary of time series'
		left = [('Begin: ',0),
					('End: ',0),
					('Number of observations: ',0),
					('Number of time series: ',0),
					('', ''),
					('Date: ', datetime.now().strftime('%a, %b %d %Y %H:%M:%S'))]

		right = [('Ann. return: ', 0),
					 ('Ann. volatility: ', 0),
					 ('Max drawdown: ', 0),
					 ('Drawdown Duration: ', 0),
					 ('Skewness: ', 0),
					 ('Kurtosis: ', 0)]
		keys = []
		values = []
		for key, value in left:
			keys.append(key)
			values.append([value])
		table = SimpleTable(values, txt_fmt=fmt_2cols, title=summary_title, stubs=keys)
		smry.tables.append(table)
		keys = []
		values = []
		for key, value in right:
			keys.append(key)
			values.append([value])
		table.extend_right(SimpleTable(values, stubs=keys))
		return smry

	def skewness(self):
		pass
	
	def kurtosis(self):
		pass
	
	def drawdown_duration(self):
		pass

	def max_drawdown(self):
		pass

	def periodic_returns(self,period):
		pass

	def rolling_max_sdrawdown(self,period=1):
		pass

	def rolling_volatility(self,lag='M'):
		"""
		Bootstrap based on blocks of the same length with end-to-start wrap around

		Parameters
		----------
		block_size : int
		Size of block to use
		args
		Positional arguments to bootstrap
		kwargs
		Keyword arguments to bootstrap

		Attributes
		----------
		index : array
		The current index of the bootstrap
		data : tuple
		Two-element tuple with the pos_data in the first position and kw_data
		in the second (pos_data, kw_data)
		pos_data : tuple
		Tuple containing the positional arguments (in the order entered)
		kw_data : dict
		Dictionary containing the keyword arguments
		random_state : RandomState
		RandomState instance used by bootstrap

		Notes
		-----
		Supports numpy arrays and pandas Series and DataFrames.  Data returned has
		the same type as the input date.

		Data entered using keyword arguments is directly accessibly as an
		attribute.

		Examples
		--------
		Data can be accessed in a number of ways.  Positional data is retained in
		the same order as it was entered when the bootstrap was initialized.
		Keyword data is available both as an attribute or using a dictionary syntax
		on kw_data.

		>>> from arch.bootstrap import CircularBlockBootstrap
		>>> from numpy.random import standard_normal
		>>> y = standard_normal((500, 1))
		>>> x = standard_normal((500, 2))
		>>> z = standard_normal(500)
		>>> bs = CircularBlockBootstrap(17, x, y=y, z=z)
		>>> for data in bs.bootstrap(100):
		...     bs_x = data[0][0]
		...     bs_y = data[1]['y']
		...     bs_z = bs.z
		"""
		returns = self.returns(period=1)

	def save(self,filename='export.csv',separator=',',date_format='%d-%m%y'):
		""" Saves the data contained in the object to a csv file.

		:param filename: path and name of the file to export
		:type filename: str, optional
		:param separator: separator between columns in file.
		:type separator: str, optional
		"""
		self.data.to_csv(filename,sep=separator,date_format='')

	def plot(self):
		""" Plots the time series."""
		self.data.plot()
		plt.show()

	def filter(self, method="L1Filter",number_simulations=100, burns=50,total_variation=2):
		""" Filters the trend of the time series.

		:param method: path and name of the file to export
		:type method: str, optional
		:param number_simulations: number of simulations in the MCMC algorithm
		:type number_simulations: int, optional
		:param burns: number of draws dismissed as burning samples
		:type burns: int, optional
		"""
		mcmc = MCMC(self, StrategyFactory.create(method,self.data.as_matrix()[:,0],total_variation_order=total_variation))
		mcmc.run(number_simulations)
		trend = mcmc.output(burns,"trend")
		self.data = self.data.join(DataFrame(trend,index=self.data.index,columns=[method]))
