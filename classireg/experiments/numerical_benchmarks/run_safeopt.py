
# Most of this code is taken from the GP classification example:
# https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_classification.ipynb

# import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)
# import matplotlib;matplotlib.rcParams['text.usetex'] = True
# import matplotlib;matplotlib.rcParams['font.size'] = 12
# import matplotlib;matplotlib.rcParams['font.family'] = 'serif'
import GPy
# from GPy.models.gp_regression import GPRegression
import numpy as np
from matplotlib import pyplot as plt
import pdb
from safeopt import SafeOpt
from safeopt import linearly_spaced_combinations

from classireg.objectives import Hartmann6D, Michalewicz10D, ConsBallRegions, Eggs2D

import torch


def get_new_evals(x,fun_obj,fun_cons):

	y_obj = -fun_obj(x,with_noise=True) # Flip sign because safeopt tries to maximize
	yl_cons = -fun_cons(x) # Flip sign because safeopt interprets g(x) >= 0 as constraint satisfaction

	y_obj = y_obj.view(-1,1).numpy()
	y_cons = yl_cons.numpy()[:,0].reshape(-1,1)

	return y_obj, y_cons


def one_experiment(my_seed,which_obj):

	torch.manual_seed(my_seed)
	np.random.seed(my_seed)

	
	if which_obj == "eggs2D":
		fun_obj = Eggs2D(noise_std=0.01)
		y_opt = 98.
		dim = 2
		x0 = torch.tensor([[0.1578, 0.558]])
		Nsamples = 100
		beta_par = 2
	elif which_obj == "hart6D":
		fun_obj = Hartmann6D(noise_std=0.01)
		y_opt = +3.32236801141551 * 10;
		dim = 6
		x0 = torch.tensor([[0.4493, 0.6189, 0.2756, 0.7961, 0.2482, 0.9121]])
		Nsamples = 15
		beta_par = 3
	else:
		raise ValueError("which_obj incorrect")
	fun_cons = ConsBallRegions(dim=dim,noise_std=0.01)

	# Initial evaluations:
	
	y0_obj, y0_cons = get_new_evals(x0,fun_obj,fun_cons)

	# Define prior:
	k_obj = GPy.kern.RBF(input_dim=dim, variance=2.0, lengthscale=0.1)
	k_cons = GPy.kern.RBF(input_dim=dim, variance=4.0, lengthscale=0.1)
	# k_obj = GPy.kern.Matern32(input_dim=dim, variance=2.0, lengthscale=0.1)
	# k_cons = GPy.kern.Matern32(input_dim=dim, variance=4.0, lengthscale=0.1)

	# Models:
	gp_obj = GPy.models.GPRegression(X=x0,Y=y0_obj,kernel=k_obj,noise_var=0.01**2)
	gp_cons = GPy.models.GPRegression(X=x0,Y=y0_cons,kernel=k_cons,noise_var=0.01**2)

	# Initialize Safeopt using the gp_classi object, i.e., a GPy.models.GPClassification model, 
	# while Safeopt actually expects a GPy.models.GPRegression model:
	bounds = [[0.0, 1.0]]*dim
	parameter_set = linearly_spaced_combinations(bounds=bounds,num_samples=Nsamples)
	opt = SafeOpt([gp_obj,gp_cons], parameter_set, fmin=[-float("Inf"), 0.], beta=beta_par)

	# print("opt.beta:",opt.beta(opt.gp.X.shape[0]))

	# pdb.set_trace()
	# gp_obj.predict_noiseless(parameter_set)

	NBOiters = 100
	y_obj = y0_obj
	y_cons = y0_cons
	x_evals = x0
	for ii in range(NBOiters):

		x_next = opt.optimize()

		x_next = torch.from_numpy(x_next).to(dtype=torch.float32)

		y_obj_next, y_cons_next = get_new_evals(x_next,fun_obj,fun_cons)

		x_evals = np.vstack((x_evals,x_next))
		y_obj = np.vstack((y_obj,y_obj_next))
		y_cons = np.vstack((y_cons,y_cons_next))

		gp_obj.set_XY(X=x_evals,Y=y_obj)
		gp_cons.set_XY(X=x_evals,Y=y_cons)


	x_best, y_obj_best = opt.get_maximum()

	regret = y_opt - y_obj_best

	return regret


if __name__ == "__main__":

	Nexp = 2
	regret_vec = np.zeros(Nexp)
	# which_obj = "eggs2D"
	which_obj = "hart6D"
	for ii in range(Nexp):
		regret_vec[ii] = one_experiment(my_seed=ii,which_obj=which_obj)
		print("regret:",regret_vec[ii])

	regret_mean = np.mean(regret_vec)
	regret_std = np.std(regret_vec)

	print("regret_mean:",regret_mean)
	print("regret_std:",regret_std)










