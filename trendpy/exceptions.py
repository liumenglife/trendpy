# -*- coding: utf-8 -*-

# exceptions.py


class NotFittedError(ValueError, AttributeError):
	pass
	
class ConvergenceWarning(UserWarning):
	pass

class FitFailedWarning(RuntimeWarning):
	pass
