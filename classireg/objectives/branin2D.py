import torch
import numpy as np
from classireg.objectives.objective_base import ObjectiveFunction
import pdb
import math

class Branin2D(ObjectiveFunction):

	def __init__(self,noise_std=0.0):
		'''
		When constraining with the circle, there's only one global minimum, and that is:
		x_opt = [ 0.5427728435726529 , 0.151666666666667 ]
		Three global minimums in the domain [0,1]x[0,1]:
		x_opt1 = [ 0.1238938230940138 , 0.818333333333333 ]
		x_opt2 = [ 0.5427728435726529 , 0.151666666666667 ]
		x_opt3 = [ 0.9616520000000001 , 0.165000000000000 ]
		'''

		super().__init__(dim=2,noise_std=noise_std)

	def evaluate(self,x_in,with_noise=False):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		# x_in = x_in.flatten()

		# Scale domain:
		x1 = 15.*x_in[:,0] - 5.
		x2 = 15.*x_in[:,1]

		# Evaluate:
		f_out = (x2 - (5.1/(4.*(math.pi)**2))*(x1)**2 + (5./math.pi)*x1 - 6.)**2 + 10.*(1.-(1./(8.*math.pi)))*torch.cos(x1) + 10.

		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		return y_out

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[ 0.534 , 0.8183 ]])
		f_gm = 0.3978872299194336
		return x_gm, f_gm


if __name__ == "__main__":

	fun = Branin2D(noise_std=0.01)

	# train_x = torch.Tensor([[1.0, 1.0]])
	train_x = torch.Tensor([[0.6255, 0.5784]])

	val = fun.evaluate(train_x)

	print("val:",val)

