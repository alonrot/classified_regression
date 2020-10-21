import torch
import numpy as np
from classireg.objectives.objective_base import ObjectiveFunction
import pdb
import math

class Shubert4D(ObjectiveFunction):

	def __init__(self,noise_std=0.0):
		'''

		http://benchmarkfcns.xyz/benchmarkfcns/shubert4fcn.html

		'''

		super().__init__(dim=4,noise_std=noise_std)

	def evaluate(self,x_in,with_noise=False):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		# x_in = x_in.flatten()

		# Scale domain:
		x_in = -2. + 4.*x_in
		assert x_in.shape[0] == 1, "Accepting only single points ..."

		# Evaluate:
		j_vec = torch.arange(1,6)

		f_out = torch.tensor([0.0])
		for k in range(self.dim):
			f_out += torch.sum( j_vec*torch.cos((j_vec + 1)*x_in[0,k] + j_vec ))

		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		return y_out

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[ 0.5 , 0.5 ]])
		f_gm = -25.74
		return x_gm, f_gm


if __name__ == "__main__":

	fun = Shubert4D()

	# train_x = torch.tensor([[1.0]*4])

	train_x = torch.tensor([[0.7162, 0.3331, 0.8390, 0.8885]]) # 11.1146

	val = fun.evaluate(train_x)

	print("val:",val)
