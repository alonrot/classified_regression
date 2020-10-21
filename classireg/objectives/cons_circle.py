import torch
from classireg.objectives.objective_base import ObjectiveFunction
import pdb
INF = float("inf")

class ConsCircle(ObjectiveFunction):

	def __init__(self,noise_std=0.0):
		'''
		g(x) = \prod_{i=1}^D sin(x_i) - t^(-D)
		This function creates 2^(D-1) disjoint unsafe areas. When t = 0, the safe/unsafe areas become perfect hypercubes.

		This function corresponds to the constraint function defined in Sec. 5.2.2 in the paper
		'''

		super().__init__(dim=2,noise_std=noise_std)

	def evaluate(self,x_in,with_noise=False):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)

		# Evaluate function noiseless
		f_out = 10*((x_in[:,0] - 0.5)**2 + (x_in[:,1] - 0.5)**2 - 2./9.)

		# Add noise:
		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		# Place -1.0 labels and INF to unstable values:
		l_out = torch.ones(len(f_out))
		l_out[y_out > 0.0] = -1
		y_out[y_out > 0.0] = INF

		return torch.cat([y_out.view(-1,1), l_out.view(-1,1)],dim=1)

if __name__ == "__main__":

	dim = 2

	cbr = ConsCircle(noise_std=0.01)

	from botorch.utils.sampling import draw_sobol_samples

	bounds = torch.Tensor([[0.0]*dim,[1.0]*dim])
	Nsamples = 20
	Nrep = 20
	x0_candidates = draw_sobol_samples(bounds=bounds,n=Nsamples,q=1).squeeze(1) # Get only unstable evaluations

	ind_stable = []
	val_list = []
	for k in range(Nsamples):

		is_stable = True
		ii = 0
		while is_stable and ii < Nrep:

			val = cbr.evaluate(x0_candidates[k,:].view(-1,dim),with_noise=True)

			is_stable = val[0,1] == +1

			ii += 1

		if ii == Nrep:
			ind_stable.append(k)
			val_list.append(val[0,0].item())


	x0_stable = x0_candidates[ind_stable,:].view(-1,dim)
	print("x0_stable:",x0_stable)
	print("val_list:",val_list)

	print("x0_candidates:",x0_candidates)
