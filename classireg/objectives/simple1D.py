import math
import torch
from .objective_base import ObjectiveFunction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class Simple1D(ObjectiveFunction):

	def __init__(self):
		'''
		'''

		super().__init__(dim=1,noise_std=0.0)

		self.x_gm = torch.tensor([[0.0]],device=device,dtype=dtype) # Dummy value

	def evaluate(self,x_in,with_noise=False):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		
		# Squeeze for 1D:
		assert x_in.shape[1] == 1, "This function is tailored for 1D cases"
		x_in = x_in.squeeze(1) # Create a 1D tensor

		# Evaluate:
		f_out = 1.0*torch.cos(x_in*6.*math.pi)*torch.exp(-x_in)

		# Add noise:
		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		return y_out

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[0.4972]],device=device,dtype=dtype)
		f_gm = -0.6074
		return x_gm, f_gm
