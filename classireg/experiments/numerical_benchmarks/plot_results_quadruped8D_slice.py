import signal
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # Solves a no-plotting issue for macOS users
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
import pdb
import sys
import yaml
from classireg.models.gpmodel import GPmodel
from classireg.models.gpcr_model import GPCRmodel
from omegaconf import DictConfig
import hydra
import torch
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
list_algo = ["EIC"]
fontsize_labels = 35
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})


def sigquit_handler(signum, frame):
	print('SIGQUIT received; exiting')
	sys.exit(os.EX_SOFTWARE)

def create_grid(dim,Ndiv):

	print("Generating the meshgrid...")
	Ndiv_vec = Ndiv*np.ones(dim,dtype=int)
	x_vec = np.linspace(0.0,1.0,Ndiv_vec[0])
	x_vec_dim = np.tile(x_vec,(dim,1))
	grid_all = np.meshgrid(*x_vec_dim,indexing='ij',sparse=False,copy=False)

	# Vectorize:
	print("Vectorizing...")
	Ntot = int(np.prod(Ndiv_vec))
	X_multidim_vec = np.atleast_2d(np.reshape(grid_all[0],Ntot))
	for k in range(1,dim):
		X_multidim_vec = np.vstack((X_multidim_vec,np.atleast_2d(np.reshape(grid_all[k],Ntot))))
	X_multidim_vec = X_multidim_vec.T

	# pdb.set_trace()

	return X_multidim_vec, grid_all

def get_posterior_mean_for_obj_and_cons(X_multidim_vec,gp_obj,gp_cons,kd_joint_min):
	"""

	This function is tailored to a specific data representaiton from experiment 20200727111325
	"""

	X_multidim_vec_proj = np.hstack([X_multidim_vec,kd_joint_min[0]*np.ones((X_multidim_vec.shape[0],1)),kd_joint_min[1]*np.ones((X_multidim_vec.shape[0],1))])


	print("Manually modifying the lengthscales (!!!!!!!!!!!!!!!!!!!!!!!!!)")
	gp_obj.covar_module.base_kernel.lengthscale = 0.1*torch.ones(4)
	gp_cons.covar_module.base_kernel.lengthscale = 0.1*torch.ones(4)

	mvn_obj = gp_obj(torch.from_numpy(X_multidim_vec_proj).to(device=device,dtype=dtype))
	mvn_cons = gp_cons(torch.from_numpy(X_multidim_vec_proj).to(device=device,dtype=dtype))

	mean_post_obj 	= mvn_obj.mean
	mean_post_cons	= mvn_cons.mean

	# Rescale *back* to physical units - Not necessary because we want to show the cost and the constraint as seen by BOC
	# mean_post_obj = (mean_post_obj - 8.0)/(-10.0)
	# mean_post_cons = mean_post_cons * 10.0 + 60.0
	mean_post_obj = (mean_post_obj - 8.0)/(-10.0) # Transform back
	mean_post_obj = -mean_post_obj # Flip sign because we wanna represent cost
	mean_post_cons = mean_post_cons * 10.0 + 60.0

	return mean_post_obj, mean_post_cons

@hydra.main(config_path="config.yaml")
def plot_GPslice(cfg: DictConfig):

	# Error checking:
	# nr_exp = "20200723160313"
	nr_exp = "20200727111325" # The one used in the paper
	which_acqui = "EIC"
	which_obj = "quadruped8D"
	if which_acqui not in list_algo:
		raise ValueError("which_acqui must be in " + str(list_algo) + ", but which_acqui: {0:s}".format(which_acqui))

	# Open corresponding file to the wanted results (we assume only one experiment has been made):
	path2data = "./{0:s}/{1:s}_results/{2:s}/data_0.yaml".format(which_obj,which_acqui,nr_exp)
	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	# pdb.set_trace()
	train_x_cons_new = torch.from_numpy(my_node["GPs"][1]['train_inputs']).to(device=device,dtype=dtype)
	train_yl_cons_new = torch.from_numpy(my_node["GPs"][1]['train_targets']).to(device=device,dtype=dtype)

	train_x_obj_new = torch.from_numpy(my_node["GPs"][0]['train_inputs']).to(device=device,dtype=dtype)
	train_y_obj_new = torch.from_numpy(my_node["GPs"][0]['train_targets']).to(device=device,dtype=dtype)

	# Create objects:
	dim = 4
	gp_obj = GPmodel(dim=dim, train_X=train_x_obj_new, train_Y=train_y_obj_new.view(-1), options=cfg.gpmodel)
	gp_cons = GPCRmodel(dim=dim, train_x=train_x_cons_new.clone(), train_yl=train_yl_cons_new.clone(), options=cfg.gpcr_model)

	Ndiv = 40
	X_multidim_vec, grid_all = create_grid(dim=2,Ndiv=Ndiv)

	# kd_joint_min_points = np.array([	[0.45684475, 0.28067630],
	# 																	[0.99251395, 0.40854543],
	# 																	[0.65000000, 0.35000000],
	# 																	[0.71000000, 0.35000000],]) # This one

	kd_joint_min_points = np.array([[0.71000000, 0.35000000]])

	# Npoints = kd_joint_min_points.shape[0]
	# hdl_fig, hdl_splot = plt.subplots(Npoints,2,figsize=(12,8))
	# hdl_splot = hdl_splot.reshape(-1,Npoints)


	figsize=(16,7)
	grid_total=(1,2)
	hdl_fig = plt.figure(figsize=figsize)
	# hdl_fig.subplots_adjust(hspace=1.0)
	hdl_splot_obj = plt.subplot2grid(grid_total, (0,0), colspan=1,fig=hdl_fig)
	hdl_splot_cons = plt.subplot2grid(grid_total, (0,1), colspan=1,fig=hdl_fig)
	# hdl_splot_evol = plt.subplot2grid(grid_total, (1,0), colspan=2,fig=hdl_fig)

	# colormap = "coolwarm"
	# colormap = "terrain"
	colormap = "rainbow"
	# color_evals_xs = "tomato"
	# color_evals_xs = "forestgreen"
	# color_evals_xs = "palegreen"
	color_evals_xs = "mediumseagreen"
	# color_evals_xu = "royalblue"
	color_evals_xu = "mediumblue"
	markersize = 14

	mean_post_obj, mean_post_cons = get_posterior_mean_for_obj_and_cons(X_multidim_vec,gp_obj,gp_cons,kd_joint_min=kd_joint_min_points[0,:])
	
	# Objective f(x):
	cset_f_obj_mean = hdl_splot_obj.contourf(grid_all[0], grid_all[1], mean_post_obj.view(Ndiv,Ndiv).detach().numpy(), 150, cmap=colormap)
	cbar_obj = hdl_fig.colorbar(cset_f_obj_mean, ax=hdl_splot_obj, ticks=np.arange(-0.8,-0.72,0.02), shrink=0.83)
	# cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
	
	hdl_splot_obj.plot(my_node["GPs"][1]['train_xs'][:,0],my_node["GPs"][1]['train_xs'][:,1],marker="o",linestyle="None",color=color_evals_xs,markersize=markersize)

	# hdl_splot_obj.set_xlabel(r"$P_{end}^{knee}$")
	# hdl_splot_obj.set_ylabel(r"$P_{end}^{hip}$")
	hdl_splot_obj.set_xlabel(r"$x_1$",fontsize=fontsize_labels)
	hdl_splot_obj.set_ylabel(r"$x_2$",fontsize=fontsize_labels)
	hdl_splot_obj.set_title(r"Cost $f(x)$",fontsize=fontsize_labels)
	hdl_splot_obj.set_xticks(np.arange(0,1.1,0.5))
	hdl_splot_obj.set_yticks(np.arange(0,1.1,0.5))
	hdl_splot_obj.set_yticklabels(["","0.5","1.0"])
	hdl_splot_obj.set_aspect('equal')

	# Constraint g(x):
	cset_f_cons_mean = hdl_splot_cons.contourf(grid_all[0], grid_all[1], mean_post_cons.view(Ndiv,Ndiv).detach().numpy(), 150, cmap=colormap)
	# cbar_cons = hdl_fig.colorbar(cset_f_cons_mean, ax=hdl_splot_cons, ticks=[-0.8, -0.76, -0.72])
	cbar_cons = hdl_fig.colorbar(cset_f_cons_mean, ax=hdl_splot_cons, ticks=np.arange(62,70.1,2), shrink=0.83)
	
	hdl_splot_cons.plot(my_node["GPs"][1]['train_xs'][:,0],my_node["GPs"][1]['train_xs'][:,1],marker="o",linestyle="None",color=color_evals_xs,markersize=markersize)
	hdl_splot_cons.plot(my_node["GPs"][1]['train_xu'][:,0],my_node["GPs"][1]['train_xu'][:,1],marker="X",linestyle="None",color=color_evals_xu,markersize=markersize)

	hdl_splot_cons.set_xlabel(r"$x_1$",fontsize=fontsize_labels)
	# hdl_splot_cons.set_ylabel(r"$x_2$",fontsize=fontsize_labels)
	hdl_splot_cons.set_title(r"Constraint $g(x)$",fontsize=fontsize_labels)
	hdl_splot_cons.set_xticks(np.arange(0,1.1,0.5))
	hdl_splot_cons.set_yticks(np.arange(0,1.1,0.5))
	hdl_splot_cons.set_yticklabels([""]*3)
	hdl_splot_cons.set_aspect('equal')

	save_plot = True
	if save_plot == True:
		print("Saving plot...")
		hdl_fig.tight_layout()
		path2save_figure = "./plots/"
		file_name = "quadruped_GPslices"
		plt.savefig(path2save_figure+file_name,dpi=300)
		print("Saved!")
	elif block == True:
		plt.show(block=True)
	else:
		plt.close(hdl_fig)




if __name__ == "__main__":

	# Handle signal that kills the program:
	signal.signal(signal.SIGQUIT, sigquit_handler)

	plot_GPslice()
