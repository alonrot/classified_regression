import torch
from torch.distributions.normal import Normal
norm_dist = Normal(loc=0.0,scale=1.0)

class ObjectiveFunction():

	def __init__(self,dim,noise_std):

		# Static attributes:
		self._dim = dim
		self._noise_std = noise_std

	@property
	def dim(self):
		return self._dim

	@property
	def noise_std(self):
		return self._noise_std

	def error_checking_x_in(self,x_in):

		x_in = x_in.view(-1,self.dim)
		assert x_in.dim() == 2, "x_in does not have the proper size"
		assert not torch.any(torch.isnan(x_in)), "x_in contains nans"
		assert not torch.any(torch.isinf(x_in)), "x_in contains Infs"
		return x_in

	def add_gaussian_noise(self,f_out):

		if isinstance(f_out,float):
			sample_shape = [1]
		else:
			sample_shape = [len(f_out)]

		return f_out + self.noise_std*norm_dist.sample(sample_shape=sample_shape)

	def __call__(self,x_in,with_noise=False):		
		y_out = self.evaluate(x_in=x_in,with_noise=with_noise)
		return y_out