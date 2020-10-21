import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import signal
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
import os
from plot_results import add_plot_attributes
from classireg.models.gpmodel import GPmodel
from classireg.models.gpcr_model import GPCRmodel
from omegaconf import DictConfig
import hydra
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

list_algo = ["EIC"]

# color_mean_dict = dict(EIC="darkorange",EI="sienna",EI_heur_high="mediumpurple",EI_heur_low="darkgreen")
color_mean_dict = dict(EIC="mediumseagreen",EI="sienna",EI_heur_high="mediumpurple",EI_heur_low="darkgreen")
marker_dict = dict(EIC="o",EI="v",EI_heur_high="+",EI_heur_low="s")
# labels_dict = dict(EIC="EIC",EI="EI",EI_heur_high="EI_heur_high",EI_heur_low="EI_heur_low")
labels_dict = dict(EIC="EIC2",EI="EI -- adaptive cost",EI_heur_high="EI -- high cost",EI_heur_low="EI -- medium cost",EIClassi="EIC with GPC")
color_bars_dict = dict(EIC="navajowhite",EI_heur_high="mediumseagreen",EIClassi="mediumpurple")
color_errorbars_dict = dict(EIC="sienna",EI_heur_high="darkgreen",EIClassi="purple")

def sigquit_handler(signum, frame):
	print('SIGQUIT received; exiting')
	sys.exit(os.EX_SOFTWARE)

def get_plotting_data_quadruped8D(nr_exp,which_acqui):

	# Error checking:
	# acqui_list = ["EIC"]
	which_obj = "quadruped8D"
	if which_acqui not in list_algo:
		raise ValueError("which_acqui must be in " + str(list_algo) + ", but which_acqui: {0:s}".format(which_acqui))

	# Open corresponding file to the wanted results (we assume only one experiment has been made):
	path2data = "./{0:s}/{1:s}_results/{2:s}/data_0.yaml".format(which_obj,which_acqui,nr_exp)
	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	# Extract relevant quantitites:
	obj_fun_train_targets = my_node["GPs"][0]['train_targets']
	cons_fun_train_targets = my_node["GPs"][1]['train_targets']
	threshold_evolution = my_node["threshold_array"]
	# pdb.set_trace()

	# # DBG:
	# print("my_node['GPs'][0]['train_inputs']")
	# print(my_node["GPs"][0]['train_inputs'])
	# print("my_node['GPs'][1]['train_inputs']")
	# print(my_node["GPs"][1]['train_inputs'])

	# Convert to meaningful values:
	obj_fun_train_targets = (obj_fun_train_targets - 8.0) / (-10.0)
	cons_fun_train_targets[:,0] = cons_fun_train_targets[:,0]*10.0 + 60.0
	threshold_evolution = threshold_evolution*10.0 + 60.0

	# Add a zero at the beginning:
	threshold_evolution = np.insert(threshold_evolution,0,0)

	obj_fun_train_targets_full = np.zeros(cons_fun_train_targets.shape[0])
	obj_fun_train_targets_full[cons_fun_train_targets[:,1] == +1] = obj_fun_train_targets[:]
	obj_fun_train_targets_full[cons_fun_train_targets[:,1] == -1] = float("Inf")

	# Further scale to cm:
	obj_fun_train_targets_full = obj_fun_train_targets_full * 100.0

	n_iters_vec = np.arange(1,cons_fun_train_targets.shape[0]+1)

	# val_cost 				= -10.0 * val_cost + 8.0
	# val_constraint	= (val_constraint-60)/10.0 # Optimistic mean

	data_for_plotting = dict(	cons_fun_train_targets=cons_fun_train_targets,
														threshold_evolution=threshold_evolution,
														obj_fun_train_targets_full=obj_fun_train_targets_full,
														n_iters_vec=n_iters_vec)

	return data_for_plotting

def plot_quadruped8D(which_obj,nr_exp,get_log_data,save_plot=True,block=True):

	marker_every = 5
	markersize = 6
	linestyle = "-"
	linewidth = 1
	fontsize_labels = 35
	xlabel_DeltaBt = "Iteration"
	xlim = [0,36]
	supress_xticks_DeltaBt = False
	matplotlib.rc('xtick', labelsize=fontsize_labels)
	matplotlib.rc('ytick', labelsize=fontsize_labels)
	matplotlib.rc('text', usetex=True)
	matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

	hdl_fig, hdl_splot = plt.subplots(2,1,figsize=(16,7))
	axis_obj_fun = hdl_splot[0]
	axis_cons_fun = hdl_splot[1]
	# axis_thres = hdl_splot[2]

	axis_obj_fun.set_ylabel("Height [cm]",fontsize=fontsize_labels)
	# axis_obj_fun.set_xticks([1+0.125/2,2+0.125/2])
	# axis_obj_fun.set_xticklabels(["Initial","Final"])
	axis_cons_fun.set_ylabel("Current [A]",fontsize=fontsize_labels)
	# axis_cons_fun.set_yticklabels([])
	# axis_cons_fun.set_yticks([])
	# axis_cons_fun.set_xticklabels([])
	# axis_cons_fun.set_xlim(xlim)
	# axis_thres = add_plot_attributes(axis_thres,fontsize_labels,r"$\hat{\varphi}^{max}$ [rad]",xlabel=xlabel_DeltaBt,supress_xticks=supress_xticks_DeltaBt)
	# axis_thres.set_xlabel("Iteration nr.",fontsize=fontsize_labels)
	# axis_thres.set_ylim([0,1.0])
	# axis_thres.set_xlim(xlim)
	

	out_dict = dict()
	legend_boxplot_list = [None]*len(list_algo)
	legend_barplot_list = [None]*len(list_algo)
	list_algo_legend_labels = [None]*len(list_algo)
	markersize = 14
	for algo in list_algo:

		dfp = get_plotting_data_quadruped8D(nr_exp,algo)
		
		print("Threshold estimated final: {0:3.4f}".format(dfp["threshold_evolution"][-1]))

		axis_obj_fun.plot(dfp["n_iters_vec"],dfp["obj_fun_train_targets_full"],marker=marker_dict[algo],label=labels_dict[algo],color=color_mean_dict[algo],linestyle="--",markersize=markersize,linewidth=2)

		axis_cons_fun.plot(dfp["n_iters_vec"],dfp["cons_fun_train_targets"][:,0],marker=marker_dict[algo],label=labels_dict[algo],color=color_mean_dict[algo],linestyle="--",markersize=markersize,linewidth=2)

		# Add red crosses:
		# pdb.set_trace()
		unstable_iters_vec = np.arange(1,dfp["cons_fun_train_targets"].shape[0]+1)[dfp["cons_fun_train_targets"][:,1] == -1]
		# axis_obj_fun.plot(unstable_iters_vec,[0.20]*len(unstable_iters_vec),marker="X",label=labels_dict[algo],color="royalblue",linestyle="None",markersize=markersize)
		# axis_cons_fun.plot(unstable_iters_vec,[60.0]*len(unstable_iters_vec),marker="X",label=labels_dict[algo],color="royalblue",linestyle="None",markersize=markersize)
		axis_obj_fun.vlines(unstable_iters_vec,linewidth=10,color="lightblue",ymin=25,ymax=80)
		axis_obj_fun.set_ylim([25,80])
		axis_cons_fun.vlines(unstable_iters_vec,linewidth=10,color="lightblue",ymin=60,ymax=120)
		axis_cons_fun.set_ylim([65,120])

		# pdb.set_trace()


		# axis_thres.plot(dfp["n_iters_vec"],dfp["threshold_evolution"],marker=marker_dict[algo],label=labels_dict[algo],color=color_mean_dict[algo],linestyle="--")

	# 	print("Initial controller performance: {0:f} ({1:f})".format(out_dict[list_algo[ii]]["init_regret_mean"],out_dict[list_algo[ii]]["init_regret_std"]))
	# 	print("Final controller performance: {0:f} ({1:f})".format(out_dict[list_algo[ii]]["final_regret_mean"],out_dict[list_algo[ii]]["final_regret_std"]))
	# 	safe_controllers = out_dict[list_algo[ii]]["safe_controllers"]
	# 	aux_nums = out_dict[list_algo[ii]]["aux_nums"]
	# 	count_single_point = out_dict[list_algo[ii]]["count_single_point"]
	# 	print("Perc. safe evaluations: {0:f} ({1:f}) -- Niters: {2:d}".format( np.mean(safe_controllers/aux_nums[-1]) , np.std(safe_controllers/aux_nums[-1]) , aux_nums[-1]))
	# 	print("Number of cases in which only one safe evaluation was found: {0:d} / {1:d}".format(count_single_point,Nrepetitions))

	# # pdb.set_trace()

	# 	legend_boxplot_list[ii] = axis_cons_fun.boxplot(x=out_dict[list_algo[ii]]["iter_num_first_stable_val_vec"],positions=[ii+1],
	# 											vert=False,whis=[15,85],patch_artist=True,widths=0.75,
	# 											flierprops=dict(markerfacecolor=color_errorbars_dict[list_algo[ii]],marker=".",markeredgecolor=color_errorbars_dict[list_algo[ii]],markersize=10),
	# 											boxprops=dict(facecolor=color_bars_dict[list_algo[ii]],color=color_bars_dict[list_algo[ii]]),
	# 											capprops=dict(color=color_bars_dict[list_algo[ii]],linewidth=2),
	# 											medianprops=dict(color=color_errorbars_dict[list_algo[ii]],linewidth=2))
	# 	legend_boxplot_list[ii] = legend_boxplot_list[ii]["boxes"][0]

	# 	mean_barplot = [ out_dict[list_algo[ii]]["init_regret_mean"] , out_dict[list_algo[ii]]["final_regret_mean"] ]
	# 	std_barplot = [ out_dict[list_algo[ii]]["init_regret_std"] , out_dict[list_algo[ii]]["final_regret_std"] ]
	# 	legend_barplot_list[ii] = axis_obj_fun.bar(x=np.array([1,2])+ii*0.125,height=mean_barplot,yerr=std_barplot,capsize=7,
	# 																							width=0.125,color=color_bars_dict[list_algo[ii]],linewidth=0,
	# 																							ecolor=color_errorbars_dict[list_algo[ii]],error_kw={"elinewidth": 2,"capthick":2})

		# if list_algo[ii] == "EIC":
		# 	print("Adding smth to the vertical axis ... (!!)")
		# 	threshold_mean = (out_dict[list_algo[ii]]["threshold_mean"] + 10.0)/10.0
		# 	# threshold_mean = out_dict[list_algo[ii]]["threshold_mean"]
		# 	threshold_std = out_dict[list_algo[ii]]["threshold_std"]/10.0
		# 	add_error_surface(mean_vec=threshold_mean,std_vec=threshold_std,axis=axis_thres,ylabel=None,color=color_mean_dict["EIC"],
		# 									label_legend=labels_dict["EIC"],marker=marker_dict["EIC"],linestyle=linestyle,linewidth=linewidth,xlabel=None,
		# 									markersize=markersize,marker_every=marker_every)

		# 	print("Last estimated threshold: {0:f} ({1:f})".format(threshold_mean[-1],threshold_std[-1]))

		# list_algo_legend_labels[ii] = labels_dict[list_algo[ii]]

	# Add legends:
	# axis_cons_fun.set_yticklabels([])
	# axis_cons_fun.set_yticks([])
	# axis_cons_fun.legend(legend_boxplot_list,list_algo_legend_labels,loc="center right",fontsize=fontsize_labels-1)
	# axis_obj_fun.legend(legend_barplot_list,list_algo_legend_labels,loc="upper center",fontsize=fontsize_labels-1)

	axis_obj_fun.set_xticklabels([""])
	axis_obj_fun.set_xlim([1,50])
	axis_obj_fun.set_xticks([1,10,20,30,40,50])
	axis_cons_fun.set_xticks([1,10,20,30,40,50])
	axis_cons_fun.set_xlim([1,50])
	axis_cons_fun.set_xlabel("Iteration nr.",fontsize=fontsize_labels)
	axis_cons_fun.set_yticks([70,90,110])


	if save_plot == True:
		print("Saving plot...")
		hdl_fig.tight_layout()
		path2save_figure = "./plots/"
		file_name = "quadruped_evolution"
		plt.savefig(path2save_figure+file_name,dpi=600)
		print("Saved!")
	elif block == True:
		plt.show(block=True)
	else:
		plt.close(hdl_fig)


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

if __name__ == "__main__":

	# Handle signal that kills the program:
	signal.signal(signal.SIGQUIT, sigquit_handler)

	if len(sys.argv) != 3:
		raise ValueError("Required input arguments: <Experiment number {202007XXXX}> <Logarithmic scale {0,1}>")
	ObjFun = "quadruped8D"
	nr_exp = sys.argv[1]
	get_log_data = sys.argv[2] == "1"

	plot_quadruped8D(which_obj=ObjFun,nr_exp=nr_exp,get_log_data=get_log_data,save_plot=False)
	# python plot_results_quadruped8D.py 20200727111325 0


	# Best guess information, obtained from exp_quadruped8D_21_07_2020.yaml :
	# height_vec = np.array([0.782,0.782,0.783,0.782,0.787,0.787,0.788,0.784,0.785])
	# height_vec_mean = np.mean(height_vec)
	# height_vec_std = np.std(height_vec)
	# print("height_vec_mean: {0:2.8f}".format(height_vec_mean))
	# print("height_vec_std: {0:2.8f}".format(height_vec_std))

	# current_vec = np.array([106.04,102.63,102.92,103.68,105.68,104.33,106.61,103.71,104.74])
	# current_vec_mean = np.mean(current_vec)
	# current_vec_std = np.std(current_vec)
	# print("current_vec_mean: {0:2.8f}".format(current_vec_mean))
	# print("current_vec_std: {0:2.8f}".format(current_vec_std))

	# height_vec_mean: 0.78444444
	# height_vec_std: 0.00226623
	# current_vec_mean: 104.48222222
	# current_vec_std: 1.31612037





