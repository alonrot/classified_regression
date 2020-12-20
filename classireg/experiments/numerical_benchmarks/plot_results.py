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
import pickle

# List of algorithms:
# list_algo = ["EIC"]
# list_algo = ["EIC_standard"]
# list_algo = ["EI","EIC"]
list_algo = ["EI_heur_low","EI_heur_high","EI","EIC","PIBU","EIC_standard"]
# list_algo = ["EI_heur_high","EIC"]
# list_algo = ["EI_heur_high","EIClassi","EIC"]

# Attributes:
# color_mean_dict = dict(EIC="goldenrod",PESC="sienna",RanCons="mediumpurple",XSF="darkgreen")
# color_mean_dict.update(dict(EI="goldenrod",PES="sienna",Ran="mediumpurple",XS="darkgreen",mES="lightcoral",PI="cornflowerblue",UCB="grey"))
# marker_dict = dict(EIC="v",PESC="o",RanCons="*",XSF="s")
# marker_dict.update(dict(EI="v",PES="o",Ran="*",XS="s",mES="D",PI="P",UCB="."))
# labels_dict = dict(EIC="EIC",PESC="PESC",RanCons="RanCons",XSF="XSF",XS="XS",EI="EI",PES="PES",Ran="Ran",mES="mES",PI="PI",UCB="UCB")

color_mean_dict = dict(EIC="darkorange",EI="sienna",EI_heur_high="mediumpurple",EI_heur_low="darkgreen")
marker_dict = dict(EIC="v",EI="o",EI_heur_high="+",EI_heur_low="s")
# labels_dict = dict(EIC="EIC",EI="EI",EI_heur_high="EI_heur_high",EI_heur_low="EI_heur_low")
labels_dict = dict(EIC="EIC2",EI="EI -- adaptive cost",EI_heur_high="EI -- high cost",EI_heur_low="EI -- medium cost",EIClassi="EIC with GPC")
color_bars_dict = dict(EIC="navajowhite",EI_heur_high="mediumseagreen",EIClassi="mediumpurple")
color_errorbars_dict = dict(EIC="sienna",EI_heur_high="darkgreen",EIClassi="purple")

def add_errorbar(mean_vec,std_vec,axis,ylabel=None,color=None,subtle=False,label_legend=None,
								plot_every=1,marker=None,linestyle=None,linewidth=1,capsize=4,marker_every=1,
								xlabel=None,markersize=5):

	# if var_vec is None:
	# 	return axis
	assert isinstance(mean_vec,np.ndarray)
	mean_vec = mean_vec.flatten()
	assert mean_vec.ndim == 1
	Npoints = mean_vec.shape[0]
	assert Npoints > 0

	if color is None:
		color = npr.uniform(size=(3,))

	# Input points:
	if plot_every is not None:
		if plot_every >= Npoints/2:
			plot_every = 1
	else:
		plot_every = 1

	x_vec = np.arange(1,Npoints+1)
	ind_plot_here = x_vec % plot_every == 0
	x_vec_sep = x_vec[ind_plot_here]
	mean_vec_sep = mean_vec[ind_plot_here]
	std_vec_sep = std_vec[ind_plot_here]

	# Plot:
	if subtle == True:
		axis.plot(x_vec,mean_vec,linestyle=":",linewidth=0.5,color=color)
	else:
		axis.errorbar(x=x_vec,y=mean_vec,yerr=std_vec/2.,marker=marker,linestyle=linestyle,
									linewidth=linewidth,color=color,errorevery=plot_every,capsize=capsize,
									markevery=marker_every,label=label_legend,markersize=markersize)
	# axis.plot(x_vec,mean_vec,marker=marker,linestyle=linestyle,linewidth=linewidth,color=color,label=label_legend)
	if xlabel is not None:
		axis.set_xlabel(xlabel)
	if ylabel is not None:
		axis.set_ylabel(ylabel)
	return axis


def add_error_surface(mean_vec,std_vec,axis,ylabel=None,color=None,label_legend=None,
											marker=None,linestyle=None,linewidth=1,xlabel=None,markersize=5,
											marker_every=5):
	
	assert isinstance(mean_vec,np.ndarray)
	mean_vec = mean_vec.flatten()
	assert mean_vec.ndim == 1
	Npoints = mean_vec.shape[0]
	assert Npoints > 0

	x_vec = np.arange(1,Npoints+1)
	fpred_quan_minus = mean_vec - std_vec
	fpred_quan_plus = mean_vec + std_vec
	# pdb.set_trace()
	axis.plot(x_vec,mean_vec,color=color,linestyle=linestyle,linewidth=linewidth,label=label_legend,marker=marker,markersize=markersize,markevery=marker_every)
	axis.fill(np.concatenate([x_vec, x_vec[::-1]]),np.concatenate([fpred_quan_minus,(fpred_quan_plus)[::-1]]),alpha=.15, fc=color, ec='None')
	if xlabel is not None:
		axis.set_xlabel(xlabel)
	if ylabel is not None:
		axis.set_ylabel(ylabel)
	return axis

def add_plot(var_vec,axis,ylabel=None,color=None,subtle=False):

	# if var_vec is None:
	# 	return axis

	assert isinstance(var_vec,np.ndarray)
	var_vec = var_vec.flatten()
	assert var_vec.ndim == 1
	Npoints = var_vec.shape[0]
	assert Npoints > 0

	if color is None:
		color = npr.uniform(size=(3,))

	# Input points:
	x_vec = range(1,Npoints+1)

	# Plot:
	if subtle == True:
		axis.plot(x_vec,var_vec,linestyle=":",linewidth=0.5,color=color)
	else:
		axis.plot(x_vec,var_vec,marker=".",linestyle="--",linewidth=1,color=color)
	# axis.set_xlabel("Nr. iters")
	if ylabel is not None:
		axis.set_ylabel(ylabel)
	return axis

def compute_mean_and_var(array_list,pop_unwanted_els=False,pop_N=None,hold_after_termination=True,NBOiters_max=None,get_log_data=False):

	Nels = len(array_list)

	array_list_copy = list(array_list)

	assert Nels > 0
	assert isinstance(array_list_copy[0],np.ndarray)
	if array_list_copy[0].ndim > 1: # Flatten
		for k in range(Nels):
			assert np.sum(array_list_copy[k].shape != 1) == 1 # Ensure 1D or 2D column or row vector
			array_list_copy[k] = array_list_copy[k].flatten()

	# Check that all dimensions are the same
	NBOiters_vec = np.zeros(Nels,dtype=int)
	for k in range(Nels):
		NBOiters_vec[k] = len(array_list_copy[k])

	if pop_unwanted_els == True:
		if pop_N is None:
			Nrequested_length = NBOiters_max
		else:
			Nrequested_length = pop_N

		N_wanted_els = np.sum(Nrequested_length == NBOiters_vec)
		assert N_wanted_els > 0, "Nrequested_length = {0:d}".format(Nrequested_length)
		c = 0
		while c < len(array_list_copy):
			if len(array_list_copy[c]) != Nrequested_length:
				array_list_copy.pop(c)
			else:
				c += 1
		assert len(array_list_copy) == N_wanted_els
	else:

		if NBOiters_max is None:
			NBOiters_max = np.max(NBOiters_vec)
		else:
			NBOiters_vec[NBOiters_vec > NBOiters_max] = NBOiters_max

		if hold_after_termination == True: # Append the last value until NBOiters_max is reached for those experiments that finished beforehand
			print("Hold after termination...")
			ind_maxBOiters_not_equal = NBOiters_max != NBOiters_vec
			if np.any(ind_maxBOiters_not_equal):
				list_pos_NBOiters_shorter = np.arange(Nels)[ind_maxBOiters_not_equal]
				for k in list_pos_NBOiters_shorter:
					# array_list_copy[k] = array_list_copy[k][0:]
					array_list_copy[k] = np.append(array_list_copy[k],np.ones(NBOiters_max-NBOiters_vec[k])*array_list_copy[k][-1])
			# else:
			# 	print("Nothing to modify...")
		else:
			if np.any(NBOiters_max != NBOiters_vec): # Cut the list
				print("NBOiters_vec:",NBOiters_vec)
				NBOiters = np.amin(NBOiters_vec)
				for k in range(Nels):
					array_list_copy[k] = array_list_copy[k][0:NBOiters]
				print("The list had to be cut from "+str(NBOiters_max)+" to "+str(NBOiters)+" !!!")
			else:
				NBOiters = NBOiters_max

	my_array = np.asarray(array_list_copy) # Vertically stacks 1-D vectors
	if NBOiters_max is not None:
		my_array = my_array[:,0:NBOiters_max]

	if get_log_data == True:
		if np.any(my_array < 0.0):
			print("<< WARNING >> Some regrets are negative (!)")
			min_value = np.min(my_array)
			my_array = my_array - min_value + 1e-4
			print("   Minimum regret across all experiments: {0:5.5f}".format(min_value))
		my_array = np.log10(my_array)

	# print("my_array:",my_array)
	mean_vec = np.mean(my_array,axis=0)
	std_vec = np.std(my_array,axis=0)

	assert mean_vec.ndim == 1
	assert std_vec.ndim == 1

	return mean_vec,std_vec

def get_plotting_data(which_obj,which_acqui,nr_exp,save_plot,block=True,pop_unwanted_els=True,NBOiters_max=None,
						get_DeltaBt=False,get_log_data=False,alternative_path=None,log_transf=False):

	# Error checking:
	# acqui_list = ["EIC"]
	if which_acqui not in list_algo:
		raise ValueError("which_acqui must be in " + str(list_algo) + ", but which_acqui: {0:s}".format(which_acqui))

	# Open corresponding file to the wanted results:
	path2data = "./{0:s}/{1:s}_results/{2:s}/data_all_exp.yaml".format(which_obj,which_acqui,nr_exp)
	if not os.path.exists(path2data):
		path2data = "./{0:s}/{1:s}_results/{2:s}/data.yaml".format(which_obj,which_acqui,nr_exp) # Backwards compatibility
	if not os.path.exists(path2data):
		pdb.set_trace()
	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	regret_simple_array_list = my_node['regret_simple_array_list']
	threshold_array_list = my_node['threshold_array_list'] if which_acqui == "EIC" else None

	# Remove Nones and infs:
	for k in range(len(regret_simple_array_list)-1,-1,-1):
		if regret_simple_array_list[k] is None or np.all(np.isinf(regret_simple_array_list[k])):
			regret_simple_array_list.pop(k)
			if threshold_array_list is not None:
				threshold_array_list.pop(k)

	Nrepetitions = len(regret_simple_array_list)

	# ==============================
	# Fix negative regret ==========
	# ==============================
	"""
	Some experiments yielded negative regret. This is because the global minimum value was exceeded due to (i) the noise in the
	sample evaluations and/or (ii) an incorrectly set global minimum value (only happened in walker). So fix this issue,
	we pre-establish a new global minimum value that lower bounds the regret obtained for a specific function objective (which_obj)
	across all repetitions and across all acquisition functions. In practice this amounts to offsetting up all
	the experiments. Since we are changing at the same time the regret of all experiments across all repetitions and 
	all acquisition functions for the same objective, this is a fair fix, and we are not favouring one method against the other.
	"""
	if which_obj == "eggs2D":
		offset_regret = 0.1
	else:
		offset_regret = 0.0

	for k in range(Nrepetitions):
		regret_simple_array_list[k] += offset_regret

	hdl_fig, hdl_splot = plt.subplots(3,1,figsize=(9,9))
	hdl_plt_regret_simple = hdl_splot[0]
	hdl_plt_threshold = hdl_splot[1]
	hdl_plt_threshold.set_xticks([])
	hdl_plt_regret_simple.set_xticks([])
	hdl_plt_regret_simple.grid(which="major",axis="both")

	# pdb.set_trace()

	for k in range(Nrepetitions):

		# Empirical Regret v1:
		if isinstance(regret_simple_array_list[k],np.ndarray) == True and regret_simple_array_list[k] is not None:
			add_plot(var_vec=regret_simple_array_list[k],axis=hdl_plt_regret_simple,ylabel="Simple Regret",subtle=True,color="cornflowerblue")

		# DeltaBt:
		if which_acqui == "EIC":
			if isinstance(threshold_array_list[k],np.ndarray) == True and threshold_array_list[k] is not None:
				add_plot(var_vec=threshold_array_list[k],axis=hdl_plt_threshold,ylabel="threshold",subtle=True,color="cornflowerblue")

			# if isinstance(rho_t_array_list[k],np.ndarray) == True and rho_t_array_list[k] is not None:
			# 	add_plot(var_vec=rho_t_array_list[k],axis=hdl_plt_delta_t,ylabel="delta_t",subtle=True,color="cornflowerblue")

	regret_simple_mean,regret_simple_std = compute_mean_and_var(regret_simple_array_list,hold_after_termination=True,NBOiters_max=NBOiters_max,get_log_data=get_log_data,pop_unwanted_els=False,pop_N=75)
	add_plot(var_vec=regret_simple_mean,axis=hdl_plt_regret_simple,color="lightcoral")

	# Add DeltaBt:
	if which_acqui == "EIC":
		if which_obj == "eggs2D":
			threshold_array_list = [np.abs(thres_array-10.0) for thres_array in threshold_array_list]
		threshold_mean, threshold_std = compute_mean_and_var(threshold_array_list,hold_after_termination=True, NBOiters_max=NBOiters_max,pop_unwanted_els=False,pop_N=75)
		add_plot(var_vec=threshold_mean,axis=hdl_plt_threshold,color="lightcoral")
	else:
		threshold_mean, threshold_std, rho_t_mean, rho_t_std = None, None, None, None

	if save_plot == True:
		print("Saving plot...")
		hdl_fig.tight_layout()
		path2save_figure = "./"
		file_name = "tmp_"+which_acqui+"_"+str(dim)
		plt.savefig(path2save_figure+file_name)
		print("Saved!")
	elif block == True:
		plt.show(block=True)
	else:
		plt.close(hdl_fig)
		return regret_simple_mean, regret_simple_std, threshold_mean, threshold_std, None, None

def add_plot_attributes(axes,fontsize_labels,ylabel,xlabel=None,supress_xticks=False):
	if xlabel is not None:
		axes.set_xlabel(xlabel,fontsize=fontsize_labels+2)
	axes.set_ylabel(ylabel,fontsize=fontsize_labels+2)
	if supress_xticks == True:
		axes.set_xticklabels([])
	else:
		axes.tick_params('x',labelsize=fontsize_labels)
	axes.tick_params('y',labelsize=fontsize_labels)
	axes.grid(b=True,which="major",color='grey', linestyle=':', linewidth=0.5)
	return axes

def get_exp_nr_most_recent(which_obj,which_acqui):

	path2data = "./{0:s}/{1:s}_results/".format(which_obj,which_acqui)

	# From all the folders that start with '2020' take the largest number (most recent):
	dir_list = os.listdir(path2data)
	name_most_recent = "0"
	for k in range(len(dir_list)):
		if "20" in dir_list[k]:
			if int(dir_list[k]) > int(name_most_recent):
				name_most_recent = dir_list[k]

	if name_most_recent == "0":
		raise ValueError("No experiment found (!)")

	nr_exp = name_most_recent

	return nr_exp

def get_exp_nr_from_file(which_obj,which_acqui):

	path2data = "./{0:s}/selector.yaml".format(which_obj)

	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	nr_exp = str(my_node["{0:s}_experiment".format(which_acqui)])

	return nr_exp

def plot(which_obj,load_from_file_selector=False,get_log_data=False,save_plot=False,block=True):

	# Get the experiment number if None passed:
	title_simp = "Simple regret"
	fontsize_labels = 18
	figsize = (9,8)

	grid_total = (3,1)
	grid_simp = grid_inf = (0,0)
	grid_threshold_inf = grid_threshold_simp = (2,0)
	# grid_delta_t_inf = grid_delta_t_simp = (3,0)
	xlabel_DeltaBt = "Iteration"
	xlabel_delta_t = "Iteration"
	supress_xticks_DeltaBt = False
	# supress_xticks_delta_t = True
	
	# General plotting settings:
	plt.rc('font', family='serif')
	plt.rc('legend',fontsize=fontsize_labels+2)

	hdl_fig_simp = plt.figure(figsize=figsize)
	hdl_splot_simp 		= plt.subplot2grid(grid_total, grid_simp, rowspan=2,fig=hdl_fig_simp)
	hdl_splot_threshold_simp = plt.subplot2grid(grid_total, grid_threshold_simp, rowspan=1,fig=hdl_fig_simp)

	hdl_splot_simp = add_plot_attributes(hdl_splot_simp,fontsize_labels,title_simp,supress_xticks=True)
	hdl_splot_threshold_simp = add_plot_attributes(hdl_splot_threshold_simp,fontsize_labels,"$\hat{c}_{opt}$",
														xlabel=xlabel_DeltaBt,supress_xticks=supress_xticks_DeltaBt)

	hdl_splot_simp.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	# hdl_splot_delta_t_simp = plt.subplot2grid(grid_total, grid_delta_t_simp, rowspan=1,fig=hdl_fig_simp)
	# hdl_splot_delta_t_simp = add_plot_attributes(hdl_splot_delta_t_simp,fontsize_labels,"$\\rho_t$",xlabel=xlabel_delta_t,
	# 												supress_xticks=supress_xticks_delta_t)
	# else:

	# 	plt.rc('font', family='serif')
	# 	plt.rc('legend',fontsize=fontsize_labels+2)
	# 	hdl_fig_simp, hdl_splot_simp = plt.subplots(1,1,sharex=False,sharey=False,figsize=figsize)
	# 	hdl_splot_simp = add_plot_attributes(hdl_splot_simp,fontsize_labels,title_simp,xlabel="Iteration")

	plot_every = 2
	marker_every = 5
	markersize = 6
	linestyle = "-"
	linewidth = 1
	capsize = 4
	str_table_list = [None]*len(list_algo)
	rescale = False

	for i in range(len(list_algo)):

		if load_from_file_selector:
			nr_exp = get_exp_nr_from_file(which_obj,list_algo[i])
		else:
			nr_exp = get_exp_nr_most_recent(which_obj,list_algo[i])

		regret_simple_mean,\
		regret_simple_std,\
		threshold_mean,\
		threshold_std,\
		rho_t_mean,\
		rho_t_std = get_plotting_data(which_obj,list_algo[i],nr_exp,save_plot=False,block=False,get_log_data=get_log_data)

		# add_errorbar(regret_simple_mean,regret_simple_std,axis=hdl_splot_simp,color=color_mean_dict[list_algo[i]],label_legend=labels_dict[list_algo[i]],
		# 							plot_every=plot_every,marker=marker_dict[list_algo[i]],linestyle=linestyle,linewidth=linewidth,capsize=capsize,marker_every=marker_every)
	

		# We can rescale all the regrets obtained by all algorithms:
		if rescale:
			init_value = regret_simple_mean[0] # The first point is assumed to be the same for all algorihtms. Hence, it equals the mean and shows no std
			# pdb.set_trace()
			if get_log_data: # xh = log(x/a) = log(x) - log(a) -> muh = mean(log(x) - log(a)) = mean(log(x)) - log(a) -> varh = var(log(x) - log(a)) = var(log(x))
				regret_simple_mean = regret_simple_mean - init_value # log(x) - log(a), where we're normalizing x/a
				# it's an "offset transformation", hence the variance oisn't affected
			else: # xh = x/a -> muh = mu/a -> varh = var/h**2 -> std = std/h
				regret_simple_mean = regret_simple_mean / init_value
				regret_simple_std = regret_simple_std / init_value 

		pdb.set_trace()

		add_error_surface(mean_vec=regret_simple_mean,std_vec=regret_simple_std,axis=hdl_splot_simp,ylabel=None,color=color_mean_dict[list_algo[i]],
											label_legend=labels_dict[list_algo[i]],marker=marker_dict[list_algo[i]],linestyle=linestyle,linewidth=linewidth,xlabel=None,
											markersize=markersize,marker_every=marker_every)

		# # if is_constrained:
		# if list_algo[i] == "EIC":
		# 	add_errorbar(threshold_mean,threshold_std,axis=hdl_splot_threshold_simp,color=color_mean_dict[list_algo[i]],label_legend=labels_dict[list_algo[i]],
		# 							plot_every=plot_every,marker=marker_dict[list_algo[i]],linestyle=linestyle,linewidth=linewidth,capsize=capsize,marker_every=marker_every)
		
		if list_algo[i] == "EIC":
			add_error_surface(mean_vec=threshold_mean,std_vec=threshold_std,axis=hdl_splot_threshold_simp,ylabel=None,color=color_mean_dict[list_algo[i]],
												label_legend=labels_dict[list_algo[i]],marker=marker_dict[list_algo[i]],linestyle=linestyle,linewidth=linewidth,xlabel=None,
												markersize=markersize,marker_every=marker_every)

	# How to change xticks:
	hdl_splot_simp.legend()

	if save_plot == True:
		print("Saving plot...")
		hdl_fig_simp.tight_layout()
		path2save_figure = "./plots/"
		file_name = "{0:s}".format(which_obj)
		plt.savefig(path2save_figure+file_name,dpi=300)
		print("Saved!")
	elif block == True:
		plt.show(block=True)
	else:
		plt.close(hdl_fig_simp)


def plot_bars(load_from_file_selector=False,get_log_data=False,save_plot=False,block=True):


	# Get the experiment number if None passed:
	fontsize_labels = 18
	figsize = (12,8)

	grid_total = (4,1)
	grid_thres = (0,0)
	grid_regret = (1,0)
	
	# General plotting settings:
	plt.rc('font', family='serif')
	plt.rc('legend', fontsize=fontsize_labels+2)

	hdl_fig = plt.figure(figsize=figsize)
	hdl_fig.subplots_adjust(hspace=1.0)
	hdl_splot_regret = plt.subplot2grid(grid_total, grid_regret, rowspan=3,fig=hdl_fig)
	hdl_splot_thres  = plt.subplot2grid(grid_total, grid_thres, rowspan=1,fig=hdl_fig, sharex=hdl_splot_regret)

	# hdl_splot_regret.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

	# plot_every = 2
	# marker_every = 5
	# markersize = 6
	# linestyle = "-"
	# linewidth = 1
	capsize = 4
	str_table_list = [None]*len(list_algo)
	rescale = False
	obj_fun_list = ["micha10D","hart6D","eggs2D"]
	# obj_fun_list = ["micha10D"]
	obj_fun_list_names = ["Michalewicz 10D","Hartman 6D","Egg crate 2D"]
	list_algo_names = ["MC+EI","HC+EI","AC+EI",r"EIC$^2$","PIBU","EIC(sta)"]
	# list_algo = ["EI_heur_low","EI_heur_high","EI","EIC","PIBU"]
	Nobj_funs = len(obj_fun_list)
	Nalgos = len(list_algo)
	regret_mean_list_algo = np.zeros((Nobj_funs,Nalgos))
	regret_std_list_algo = np.zeros((Nobj_funs,Nalgos))
	thres_mean_list_objs = np.zeros(Nobj_funs)
	thres_std_list_objs = np.zeros(Nobj_funs)
	thres_mean_EIC_obj = thres_std_EIC_obj = 0.0
	# Ncut = 60
	Ncut = 100
	for j in range(Nobj_funs):

		for i in range(Nalgos):


			if list_algo[i] == "PIBU":

				path2mat_data = "/Users/alonrot/MPI/WIP_projects/pibu/results/{0:s}/regret_data.pkl".format(obj_fun_list[j])
				with open(path2mat_data, 'rb') as fin :
					regret_data = pickle.load(fin)
					regret_data = regret_data.squeeze()

					# pdb.set_trace()
					regret_simple_mean = regret_data[0]
					regret_simple_std = regret_data[1]

				regret_mean_list_algo[j,i] = regret_simple_mean
				regret_std_list_algo[j,i] = regret_simple_std

			else:

				if load_from_file_selector:
					nr_exp = get_exp_nr_from_file(obj_fun_list[j],list_algo[i])
				else:
					nr_exp = get_exp_nr_most_recent(obj_fun_list[j],list_algo[i])

				regret_simple_mean,\
				regret_simple_std,\
				threshold_mean,\
				threshold_std,\
				rho_t_mean,\
				rho_t_std = get_plotting_data(obj_fun_list[j],list_algo[i],nr_exp,save_plot=False,block=False,get_log_data=get_log_data)


				regret_mean_list_algo[j,i] = regret_simple_mean[Ncut-1]
				regret_std_list_algo[j,i] = regret_simple_std[Ncut-1]

			if list_algo[i] == "EIC": # EIC is actually EIC2
				thres_mean_EIC_obj = threshold_mean[Ncut-1]
				thres_std_EIC_obj = threshold_std[Ncut-1]

				# if obj_fun_list[j] == "eggs2D": # This isn't necessary because in get_plotting_data() we have already subtracted 10.0: line 281-282 (commented out by default; uncomment for obatining the desired plot)
				# 	thres_mean_EIC_obj -= 10.0

		thres_mean_list_objs[j] = thres_mean_EIC_obj
		thres_std_list_objs[j] = thres_std_EIC_obj


	# We need to normalize the regret across objectives !!!!
	# hdl_splot_regret.plot()

	pdb.set_trace()

	# Normalize:
	norm_const_vec = 1.1*np.amax(regret_mean_list_algo,axis=1)
	regret_mean_list_algo = regret_mean_list_algo / norm_const_vec.reshape(-1,1)
	regret_std_list_algo = regret_std_list_algo / norm_const_vec.reshape(-1,1)
	regret_std_list_algo = regret_std_list_algo / 4.0

	# pdb.set_trace()

	xx = np.arange(Nobj_funs)  # the label locations
	width = 0.15  # the width of the bars

	# Colors:
	color_list = plt.cm.Set2(np.arange(Nalgos-1,-1,-1))

	if Nalgos > 1:
		bar_center_points = np.linspace(-(Nalgos-1)*width/2,+(Nalgos-1)*width/2,Nalgos)
	else:
		bar_center_points = [0]



	# Plot regret:
	for ii in range(Nalgos):
		hdl_splot_regret.bar(xx + bar_center_points[ii], regret_mean_list_algo[:,ii], width, label=list_algo_names[ii], yerr=regret_std_list_algo[:,ii], capsize=5.0,color=color_list[ii,:])
	# hdl_splot_regret.bar(xx - width/2, regret_mean_list_algo[:,1], width, label=list_algo_names[1], yerr=regret_std_list_algo[:,1], capsize=5.0,color=color_list[1,:])
	# hdl_splot_regret.bar(xx + width/2, regret_mean_list_algo[:,2], width, label=list_algo_names[2], yerr=regret_std_list_algo[:,2], capsize=5.0,color=color_list[2,:])
	# hdl_splot_regret.bar(xx + width/2*3, regret_mean_list_algo[:,3], width, label=list_algo_names[3], yerr=regret_std_list_algo[:,3], capsize=5.0,color=color_list[3,:])

	# Add tick labels:
	hdl_splot_regret.set_xticks(xx)
	hdl_splot_regret.set_xticklabels(obj_fun_list_names, fontsize=fontsize_labels)
	hdl_splot_regret.tick_params(axis='y', labelsize=fontsize_labels)

	hdl_splot_regret.set_ylim([0.0,1.50])
	hdl_splot_regret.set_ylabel(r"Simple regret $r_T$", fontsize=fontsize_labels)
	
	hdl_splot_regret.legend(loc="upper center",ncol=Nalgos)
	hdl_splot_regret.grid(True,which="major",axis="y",linestyle="--")
	hdl_splot_regret.set_axisbelow(True)

	# Plot threshold:
	hdl_splot_thres.bar(xx, thres_mean_list_objs, width, align='center', tick_label=[""]*Nobj_funs, yerr=thres_std_list_objs, capsize=5.0, color=color_list[Nalgos-1,:])
	# hdl_splot_thres.axhline(y=0.0, color="black", linewidth=1)
	hdl_splot_thres.set_ylabel(r"Threshold $\hat{c}$", fontsize=fontsize_labels)
	hdl_splot_thres.set_xticks(xx)
	hdl_splot_thres.set_xticklabels(obj_fun_list_names, fontsize=fontsize_labels)
	hdl_splot_thres.tick_params(axis='y', labelsize=fontsize_labels)
	hdl_splot_thres.set_ylim([0.0,0.3])
	hdl_splot_thres.set_yticks([0.0,0.10,0.20,0.30])
	hdl_splot_thres.grid(True,which="major",axis="y",linestyle="--")
	hdl_splot_thres.set_axisbelow(True)


	if save_plot == True:
		print("Saving plot...")
		hdl_fig.tight_layout()
		path2save_figure = "./plots/"
		file_name = "barplot_regrets_all"
		plt.savefig(path2save_figure+file_name,dpi=600)
		print("Saved!")
	elif block == True:
		plt.show(block=True)
	else:
		plt.close(hdl_fig)




def get_plotting_data_walker(load_from_file_selector,which_acqui):

	which_obj = "walker"
	if load_from_file_selector:
		nr_exp = get_exp_nr_from_file(which_obj,which_acqui)
	else:
		nr_exp = get_exp_nr_most_recent(which_obj,which_acqui)

	# Error checking:
	# acqui_list = ["EIC"]
	if which_acqui not in list_algo:
		raise ValueError("which_acqui must be in " + str(list_algo) + ", but which_acqui: {0:s}".format(which_acqui))

	# Open corresponding file to the wanted results:
	path2data = "./{0:s}/{1:s}_results/{2:s}/data.yaml".format(which_obj,which_acqui,nr_exp)
	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	regret_simple_array_list = my_node['regret_simple_array_list'].copy()
	threshold_array_list = my_node['threshold_array_list'] if which_acqui == "EIC" else None

	# pdb.set_trace()

	# Remove Nones and infs:
	Nrepetitions_init = len(regret_simple_array_list)
	for k in range(len(regret_simple_array_list)-1,-1,-1):
		if regret_simple_array_list[k] is None or np.all(np.isinf(regret_simple_array_list[k])):
			regret_simple_array_list.pop(k)
			if threshold_array_list is not None:
				threshold_array_list.pop(k)

	# Pop inf thresholds:
	if which_acqui == "EIC":
		for k in range(len(threshold_array_list)-1,-1,-1):
			if np.all(np.isinf(threshold_array_list[k])):
				threshold_array_list.pop(k)

	# Remove initial unstable values:
	# pdb.set_trace()
	init_unstable_value_for_EI_heur_high = 24.0
	if which_acqui == "EI_heur_high":
		for k in range(len(regret_simple_array_list)-1,-1,-1):
			if np.all(regret_simple_array_list[k] == init_unstable_value_for_EI_heur_high):
				regret_simple_array_list.pop(k)

	# pdb.set_trace()

	# # Remove initial unstable values:
	# init_unstable_value_for_EI_heur_high = 24.0
	# if which_acqui == "EI_heur_high":
	# 	for k in range(len(regret_simple_array_list)-1,-1,-1):
	# 		if np.all(regret_simple_array_list[k] == init_unstable_value_for_EI_heur_high):
	# 			regret_simple_array_list.pop(k)

	Nrepetitions = len(regret_simple_array_list)
	print("Popped out {0:d} / {1:d} regret results for being all Infs or NaNs. This happens when the optimizer cannot\n \
				find a single safe area. We account for these statistics by explicitly mentioning the \% of failures.".format(Nrepetitions_init-Nrepetitions,Nrepetitions_init))

	# Count cases where only one safe controller was discovered:
	count_single_point = 0
	for k in range(len(regret_simple_array_list)-1,-1,-1):
		if which_acqui == "EI_heur_high":
			ind_regret_finite = regret_simple_array_list[k] != init_unstable_value_for_EI_heur_high
		elif which_acqui == "EIC":
			ind_regret_finite = regret_simple_array_list[k] != float("Inf")
		elif which_acqui == "EIClassi":
			ind_regret_finite = regret_simple_array_list[k] != float("Inf")

	# if which_acqui == "EIClassi":
	# 	pdb.set_trace()



		# pdb.set_trace()
		if np.all(np.diff(regret_simple_array_list[k][ind_regret_finite]) == 0.0): # If all are equal remove them
			count_single_point += 1

	# pdb.set_trace()

	# Get the iteration at which the first stable evaluation occured:
	iter_num_first_stable_val_vec = np.zeros(Nrepetitions)
	init_regret = np.zeros(Nrepetitions)
	final_regret = np.zeros(Nrepetitions)
	safe_controllers = np.zeros(Nrepetitions)
	for k in range(Nrepetitions):
		aux_nums = np.arange(1,len(regret_simple_array_list[k])+1)
		if which_acqui == "EI_heur_high":
			ind_regret_finite = regret_simple_array_list[k] != init_unstable_value_for_EI_heur_high
		elif which_acqui == "EIC":
			ind_regret_finite = regret_simple_array_list[k] != float("Inf")
		elif which_acqui == "EIClassi":
			ind_regret_finite = regret_simple_array_list[k] != float("Inf")
		iter_num_first_stable_val_vec[k] = aux_nums[ind_regret_finite][0] # among all the infinities, take the first index
		init_regret[k] = regret_simple_array_list[k][ind_regret_finite][0]
		final_regret[k] = regret_simple_array_list[k][ind_regret_finite][-1]
		safe_controllers[k] = np.sum(ind_regret_finite)

	# init_regret = np.log(init_regret)
	# final_regret = np.log(final_regret)

	init_regret_mean = np.mean(init_regret)
	final_regret_mean = np.mean(final_regret)
	init_regret_std = np.std(init_regret)
	final_regret_std = np.std(final_regret)

	# hdl_fig, hdl_splot = plt.subplots(2,1)
	# hdl_splot[0].hist(init_regret)
	# hdl_splot[1].hist(final_regret)
	# plt.show(block=True)

	if which_acqui == "EIC":
		threshold_mean, threshold_std = compute_mean_and_var(threshold_array_list,hold_after_termination=True,pop_unwanted_els=False,pop_N=75)
		if np.any(np.isnan(threshold_mean)) or np.any(np.isinf(threshold_mean)):
			pdb.set_trace()
	else:
		threshold_mean, threshold_std = None, None


	print("iter_num_first_stable_val_vec:",iter_num_first_stable_val_vec)
	Nmin_iters = 20
	Ncases_first_stable_obs_found_after_XXiters = np.sum(iter_num_first_stable_val_vec > Nmin_iters)

	# Inform about the best controller found:
	if "train_ys_list" in my_node.keys() and which_acqui == "EIC":

		train_xs_list = my_node["train_xs_list"]
		train_ys_list = my_node["train_ys_list"]
		regret_simple_array_list_new = my_node['regret_simple_array_list']
		y_best = float("Inf")
		for k in range(Nrepetitions_init):
			if regret_simple_array_list_new[k] is not None and not np.all(np.isinf(regret_simple_array_list_new[k])) and train_ys_list[k] is not None:
				ind_best = np.argmin(train_ys_list[k])
				y_best_new = train_ys_list[k][ind_best]
				try:
					if y_best_new < y_best:
						x_best = train_xs_list[k][ind_best,:]
						y_best = y_best_new
				except:
					pdb.set_trace()

		print("<< Best controller among all experiments >>")
		print("x_best:",x_best)
		print("y_best:",y_best)


	return init_regret_mean, final_regret_mean, init_regret_std, final_regret_std, threshold_mean, \
				threshold_std, Nmin_iters, Ncases_first_stable_obs_found_after_XXiters, Nrepetitions, \
				iter_num_first_stable_val_vec, safe_controllers, aux_nums, count_single_point

def plot_walker(load_from_file_selector,save_plot,block=True):

	marker_every = 5
	markersize = 6
	linestyle = "-"
	linewidth = 1
	fontsize_labels = 14
	xlabel_DeltaBt = "Iteration"
	xlim = [0,100]
	supress_xticks_DeltaBt = False
	matplotlib.rc('xtick', labelsize=fontsize_labels)
	matplotlib.rc('ytick', labelsize=fontsize_labels)
	matplotlib.rc('text', usetex=True)
	matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

	hdl_fig, hdl_splot = plt.subplots(3,1,figsize=(7,7))
	axis_regrets = hdl_splot[0]
	axis_initial_evs = hdl_splot[1]
	axis_thres = hdl_splot[2]

	axis_regrets.set_ylabel("Regret",fontsize=fontsize_labels)
	axis_regrets.set_xticks([1+0.125/2,2+0.125/2])
	axis_regrets.set_xticklabels(["Initial","Final"])
	axis_initial_evs.set_ylabel("Initial safe \nevaluations",fontsize=fontsize_labels)
	axis_initial_evs.set_yticklabels([])
	axis_initial_evs.set_yticks([])
	# axis_initial_evs.set_xticklabels([])
	axis_initial_evs.set_xlim(xlim)
	axis_thres = add_plot_attributes(axis_thres,fontsize_labels,r"$\hat{\varphi}^{max}$ [rad]",xlabel=xlabel_DeltaBt,supress_xticks=supress_xticks_DeltaBt)
	axis_thres.set_xlabel("Iteration nr.",fontsize=fontsize_labels)
	axis_thres.set_ylim([0,1.0])
	axis_thres.set_xlim(xlim)
	
	out_dict = dict()
	legend_boxplot_list = [None]*len(list_algo)
	legend_barplot_list = [None]*len(list_algo)
	list_algo_legend_labels = [None]*len(list_algo)
	for ii in range(len(list_algo)):

		out_keys = ["init_regret_mean","final_regret_mean","init_regret_std","final_regret_std","threshold_mean","threshold_std",
								"Nmin_iters", "Ncases_first_stable_obs_found_after_XXiters", "Nrepetitions","iter_num_first_stable_val_vec","safe_controllers",
								"aux_nums","count_single_point"]
		out_vals = get_plotting_data_walker(load_from_file_selector,list_algo[ii])

		out_dict.update( {list_algo[ii]: dict(zip(out_keys, out_vals))} )

		print("Acquisition: {0:s}".format(list_algo[ii]))
		Nmin_iters = out_dict[list_algo[ii]]["Nmin_iters"]
		Ncases_first_stable_obs_found_after_XXiters = out_dict[list_algo[ii]]["Ncases_first_stable_obs_found_after_XXiters"]
		Nrepetitions = out_dict[list_algo[ii]]["Nrepetitions"]
		print("Number of cases in which the first stable observation was found after {0:d} iterations: {1:d} / {2:d}".format(Nmin_iters,Ncases_first_stable_obs_found_after_XXiters,Nrepetitions))


		print("Initial controller performance: {0:f} ({1:f})".format(out_dict[list_algo[ii]]["init_regret_mean"],out_dict[list_algo[ii]]["init_regret_std"]))
		print("Final controller performance: {0:f} ({1:f})".format(out_dict[list_algo[ii]]["final_regret_mean"],out_dict[list_algo[ii]]["final_regret_std"]))
		safe_controllers = out_dict[list_algo[ii]]["safe_controllers"]
		aux_nums = out_dict[list_algo[ii]]["aux_nums"]
		count_single_point = out_dict[list_algo[ii]]["count_single_point"]
		print("Perc. safe evaluations: {0:f} ({1:f}) -- Niters: {2:d}".format( np.mean(safe_controllers/aux_nums[-1]) , np.std(safe_controllers/aux_nums[-1]) , aux_nums[-1]))
		print("Number of cases in which only one safe evaluation was found: {0:d} / {1:d}".format(count_single_point,Nrepetitions))

	# pdb.set_trace()

		legend_boxplot_list[ii] = axis_initial_evs.boxplot(x=out_dict[list_algo[ii]]["iter_num_first_stable_val_vec"],positions=[ii+1],
												vert=False,whis=[15,85],patch_artist=True,widths=0.75,
												flierprops=dict(markerfacecolor=color_errorbars_dict[list_algo[ii]],marker=".",markeredgecolor=color_errorbars_dict[list_algo[ii]],markersize=10),
												boxprops=dict(facecolor=color_bars_dict[list_algo[ii]],color=color_bars_dict[list_algo[ii]]),
												capprops=dict(color=color_bars_dict[list_algo[ii]],linewidth=2),
												medianprops=dict(color=color_errorbars_dict[list_algo[ii]],linewidth=2))
		legend_boxplot_list[ii] = legend_boxplot_list[ii]["boxes"][0]

		mean_barplot = [ out_dict[list_algo[ii]]["init_regret_mean"] , out_dict[list_algo[ii]]["final_regret_mean"] ]
		std_barplot = [ out_dict[list_algo[ii]]["init_regret_std"] , out_dict[list_algo[ii]]["final_regret_std"] ]
		legend_barplot_list[ii] = axis_regrets.bar(x=np.array([1,2])+ii*0.125,height=mean_barplot,yerr=std_barplot,capsize=7,
																								width=0.125,color=color_bars_dict[list_algo[ii]],linewidth=0,
																								ecolor=color_errorbars_dict[list_algo[ii]],error_kw={"elinewidth": 2,"capthick":2})

		if list_algo[ii] == "EIC":
			print("Adding smth to the vertical axis ... (!!)")
			threshold_mean = (out_dict[list_algo[ii]]["threshold_mean"] + 10.0)/10.0
			# threshold_mean = out_dict[list_algo[ii]]["threshold_mean"]
			threshold_std = out_dict[list_algo[ii]]["threshold_std"]/10.0
			add_error_surface(mean_vec=threshold_mean,std_vec=threshold_std,axis=axis_thres,ylabel=None,color=color_mean_dict["EIC"],
											label_legend=labels_dict["EIC"],marker=marker_dict["EIC"],linestyle=linestyle,linewidth=linewidth,xlabel=None,
											markersize=markersize,marker_every=marker_every)

			print("Last estimated threshold: {0:f} ({1:f})".format(threshold_mean[-1],threshold_std[-1]))

		list_algo_legend_labels[ii] = labels_dict[list_algo[ii]]

	# Add legends:
	axis_initial_evs.set_yticklabels([])
	axis_initial_evs.set_yticks([])
	axis_initial_evs.legend(legend_boxplot_list,list_algo_legend_labels,loc="center right",fontsize=fontsize_labels-1)
	axis_regrets.legend(legend_barplot_list,list_algo_legend_labels,loc="upper center",fontsize=fontsize_labels-1)


	if save_plot == True:
		print("Saving plot...")
		hdl_fig.tight_layout()
		path2save_figure = "./plots/"
		file_name = "walker"
		plt.savefig(path2save_figure+file_name,dpi=300)
		print("Saved!")
	elif block == True:
		plt.show(block=True)
	else:
		plt.close(hdl_fig)


if __name__ == "__main__":

	if len(sys.argv) != 4:
		raise ValueError("Required input arguments: <ObjFun {hart6D,branin2D}> <Load from selector.yaml {0,1}> <Logarithmic scale {0,1}>")
	ObjFun = sys.argv[1]
	load_from_file_selector = sys.argv[2] == "1"
	get_log_data = sys.argv[3] == "1"

	if ObjFun == "walker":
		plot_walker(load_from_file_selector=load_from_file_selector,save_plot=False)
	else:
		# plot(which_obj=ObjFun,load_from_file_selector=load_from_file_selector,get_log_data=get_log_data,save_plot=False)
		plot_bars(load_from_file_selector=load_from_file_selector,get_log_data=get_log_data,save_plot=False)





