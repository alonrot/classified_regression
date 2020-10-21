import torch
import numpy as np
from classireg.objectives.objective_base import ObjectiveFunction
import pdb

class Hartmann6D(ObjectiveFunction):

	def __init__(self,noise_std=0.0):
		'''
    Global minimum reported in the GOBench
    ======================================
    self.global_optimum = [[0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]]
    self.fglob = -3.32236801141551
    domain = [0,1]^D

		Implementation following GOBench
		================================
		See https://github.com/philipmorrisintl/GOBench/blob/master/gobench/go_benchmark_functions/go_funcs_H.py

		Equations
		=========
		A Literature Survey of Benchmark Functions For Global Optimization Problems
		https://arxiv.org/pdf/1308.4008.pdf

		Visualization (reported global minima unreliable...)
		=============
		http://infinity77.net/global_optimization/test_functions.html#test-functions-index

		'''

		super().__init__(dim=6,noise_std=noise_std)
		

		self.a = torch.tensor([[10., 3., 17., 3.5, 1.7, 8.],
												[0.05, 10., 17., 0.1, 8., 14.],
												[3., 3.5, 1.7, 10., 17., 8.],
												[17., 8., 0.05, 10., 0.1, 14.]])

		self.p = torch.tensor([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
												[0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
												[0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
												[0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

		self.c = torch.tensor([1.0, 1.2, 3.0, 3.2])

		self.x_gm = np.array([[0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]])

	def evaluate(self,x_in,with_noise=False):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		assert x_in.shape[0] == 1

		d = torch.sum(self.a * (x_in.repeat(4,1) - self.p) ** 2, axis=1)
		f_out = -torch.sum(self.c * torch.exp(-d)).view(1)

		# Scaling:
		# f_out += 3.32236801141551
		f_out *= 10.0

		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		return y_out

	@staticmethod
	def true_minimum():
		x_gm = np.array([[0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]])
		f_gm = -3.32236801141551 * 10
		# f_gm = 0.0 # Due to the scaling, this is the new minimum
		return x_gm, f_gm

if __name__ == "__main__":

	hartfun = Hartmann6D(noise_std=0.01)

	# train_x = torch.Tensor([[1.0, 1.0]])
	train_x = torch.Tensor([[0.4493, 0.6189, 0.2756, 0.7961, 0.2482, 0.9121]])

	val = hartfun.evaluate(train_x)

	print("val:",val)

