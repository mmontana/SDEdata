import numpy as np

class BaseModel(object):
		
	self.dim = None	
	self.integration_step = None
	self.params = dict()

	#set all vars as properties

	def __init__(self, x_init):
		""" """
		#dimensionality check		
		if x_init.shape[1] != self.dim:
			#TODO raise error here 
			pass
		self._traj = np.array([x_init])

	def integrate(tau, length):
		#TODO

	
class UserModel(object):
	""" """
	pass
