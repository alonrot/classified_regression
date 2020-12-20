import torch
import numpy as np
from classireg.objectives.objective_base import ObjectiveFunction
import pdb
import math

class Eggs2D(ObjectiveFunction):

	def __init__(self,noise_std=0.0):
		'''

		http://benchmarkfcns.xyz/benchmarkfcns/eggcratefcn.html

		'''

		super().__init__(dim=2,noise_std=noise_std)

	def evaluate(self,x_in,with_noise=False):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		# x_in = x_in.flatten()

		# Scale domain:
		x1 = -5. + 7.5*x_in[:,0]
		x2 = -2.5 + 7.5*x_in[:,1]

		# Evaluate:
		f_out = x1**2 + x2**2 + 25. * (torch.sin(x1)**2 + torch.sin(x2)**2)

		f_out -= 98.

		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		return y_out

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[ 2./3 , 1./3 ]])
		f_gm = -98. # After scaling
		return x_gm, f_gm


if __name__ == "__main__":

	fun = Eggs2D()

	train_x = torch.Tensor([[2./3, 1./3]])
	# train_x = torch.Tensor([[0.5578, 0.0558]])

	# train_x = torch.Tensor([[0.0, 1.0]])

	val = fun.evaluate(train_x)

	print("val:",val)

