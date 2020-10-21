import torch
import numpy as np
from classireg.objectives.objective_base import ObjectiveFunction
import pdb
import math

class Camel2D(ObjectiveFunction):

	def __init__(self,noise_std=0.0):
		'''

		http://benchmarkfcns.xyz/benchmarkfcns/threehumpcamelfcn.html
		https://arxiv.org/pdf/1308.4008.pdf

		'''

		super().__init__(dim=2,noise_std=noise_std)

	def evaluate(self,x_in,with_noise=False):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		# x_in = x_in.flatten()

		# Scale domain:
		x1 = -4. + 8.*x_in[:,0]
		x2 = -4. + 8.*x_in[:,1] 

		# Evaluate:

		f_out = 2*x1**2 - 1.05*x1**4 + (1./6)*x1**6 + x1*x2 + x2**2

		f_out -= 480.

		f_out = f_out / 5.

		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		return y_out

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[ 0.5 , 0.5 ]])
		# f_gm = 0.0
		f_gm = -480./5. # After rescaling
		return x_gm, f_gm


if __name__ == "__main__":

	fun = Camel2D()

	train_x = torch.tensor([[0.9846, 0.0587]])
	# train_x = torch.tensor([[0.3602, 0.2062]])
	# train_x = torch.Tensor([[0.1285, 0.7559]])
	# train_x = torch.Tensor([[0.5, 0.5]])
	# train_x = torch.Tensor([[0.5, 0.5]])

	val = fun.evaluate(train_x)

	print("val:",val)
