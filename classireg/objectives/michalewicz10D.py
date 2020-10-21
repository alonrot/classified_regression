import torch
import numpy as np
import math
from classireg.objectives.objective_base import ObjectiveFunction
import pdb

class Michalewicz10D(ObjectiveFunction):

	def __init__(self,noise_std=0.0):
		'''
		Reported global minimum
		=======================
		Vanaret, Charlie, Jean-Baptiste Gotteland, Nicolas Durand, and Jean-Marc Alliot. 
		"Certified global minima for a benchmark of difficult optimization problems." (2014).
		x_gm = np.array([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796]])
		f_gm = -9.6601517
		domain = [0,pi]^D

		In this paper, the Michalewicz1 parameter is set to m=10, like in our case

		Implementation following GOBench
		================================
		See https://github.com/philipmorrisintl/GOBench/blob/master/gobench/go_benchmark_functions/go_funcs_M.py

		Equations
		=========
		A Literature Survey of Benchmark Functions For Global Optimization Problems
		https://arxiv.org/pdf/1308.4008.pdf

		Visualization (reported global minima unreliable...)
		=============
		http://infinity77.net/global_optimization/test_functions.html#test-functions-index

		'''

		super().__init__(dim=10,noise_std=noise_std)

		self.m = 10 # Speed of oscilations
		self.x_gm = torch.tensor([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796]]) / math.pi

	def evaluate(self,x_in,with_noise=False):
		'''
		Overrride to allow for more than single-point evaluation
		'''

		# pdb.set_trace()

		x_in = self.error_checking_x_in(x_in)
		assert x_in.shape[0] == 1
		x_in = x_in.flatten()

		# pdb.set_trace()

		x_in = x_in * math.pi # Domain x_in \in [0,pi]^D

		i = torch.arange(1, self.dim + 1)
		aux_vec = torch.sin(x_in) * torch.sin((x_in ** 2)*i / math.pi) ** (2 * self.m)
		if aux_vec.dim() == 1:
			f_out = -torch.sum(aux_vec).view(1)
		else:
			f_out = -torch.sum(aux_vec,axis=1)

		# # Normalize:
		# f_out = f_out / 9.6601517 + 0.5 # Normalized to have zero mean and unit variance

		# f_out *= 5.0

		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		return y_out

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796]]) / math.pi
		f_gm = -9.6601517
		return x_gm, f_gm

if __name__ == "__main__":

	fun = Michalewicz10D()

	train_x = torch.Tensor([[0.7139, 0.6342, 0.2331, 0.8299, 0.7615, 0.8232, 0.9008, 0.1899, 0.6961, 0.3240]]) # good
	# train_x = torch.Tensor([[0.1682, 0.8252, 0.1863, 0.7816, 0.1167, 0.1846, 0.7365, 0.4733, 0.6546, 0.6552]]) # good
	# train_x = torch.Tensor([[0.1038, 0.8883, 0.2480, 0.7668, 0.6919, 0.4371, 0.8991, 0.6679, 0.0778,0.3226]])
	# train_x = torch.Tensor([[0.65456088, 0.22632844, 0.50252072, 0.80747863, 0.11509346, 0.73440179, 0.06093292, 0.464906, 0.01544494, 0.90179168]]) # Randomly computed

	val = fun.evaluate(train_x)

	print("val:",val)
