import pdb
import numpy as np
import yaml
from classireg.objectives.walker import WalkerObj, WalkerCons, one_eval
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


def main():
	"""

	We want to show what it takes to pre-compute the constraint threshold. For this we need to:
	1) Obtain a stable controller
	2) Evaluate such controller a number of times (say N=20) and, each time, observe the constraint value,
			which will be a real value (in the case of walker, the max. tilting angle).
	3) Take the mean and std

	We need to asses the above thing statistically. To this end, the initial stable controller
	is obtained by, first running EI 100 times on the real robot; second storing the FIRST controller
	of each run that stabilized the system; third, out of all those FIRST controllers, select
	the one that had the lowest regret. Then, use this one as the "stable controller" mentioned in 
	step 1) above.

	We do not need to re-run EI 100 times, we already did this. So, we take up a batch of existing experiments, 
	in this case ./walker/EI_heur_high_results/20200610004246
	This is the same batch that se use in Fig. 4 in the paper.

	"""

	which_obj = "walker"
	which_acqui = "EI_heur_high"
	exp_nr = "20200610004246"

	# Open corresponding file to the wanted results:
	path2data = "./{0:s}/{1:s}_results/{2:s}/data.yaml".format(which_obj,which_acqui,exp_nr)
	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	regret_simple_array_list = my_node['regret_simple_array_list'].copy()
	threshold_array_list = my_node['threshold_array_list'] if which_acqui == "EIC" else None
	train_x_list = my_node['train_x_list']
	train_y_list = my_node['train_y_list']

	# Remove Nones and infs:
	Nrepetitions_init = len(regret_simple_array_list)
	for k in range(len(regret_simple_array_list)-1,-1,-1):
		if regret_simple_array_list[k] is None or np.all(np.isinf(regret_simple_array_list[k])):
			regret_simple_array_list.pop(k)
			if threshold_array_list is not None:
				threshold_array_list.pop(k)
			if train_x_list is not None:
				train_x_list.pop(k)
			if train_y_list is not None:
				train_y_list.pop(k)

	# pdb.set_trace()

	# Remove initial unstable values:
	init_unstable_value_for_EI_heur_high = 0.0
	if which_acqui == "EI_heur_high":
		for k in range(len(train_y_list)-1,-1,-1):
			# if np.all(regret_simple_array_list[k] == init_unstable_value_for_EI_heur_high):
			# 	regret_simple_array_list.pop(k)
			# if train_x_list is not None:
			if np.all(train_y_list[k] == 0):
				train_x_list.pop(k)
				train_y_list.pop(k)

	# Nrepetitions = len(regret_simple_array_list)
	Nrepetitions = len(train_y_list)
	print("Popped out {0:d} / {1:d} regret results for being all Infs or NaNs. This happens when the optimizer cannot\n \
				find a single safe area. We account for these statistics by explicitly mentioning the \% of failures.".format(Nrepetitions_init-Nrepetitions,Nrepetitions_init))

	# pdb.set_trace()
	# Get the iteration at which the first stable evaluation occured:
	iter_num_first_stable_val_vec = np.zeros(Nrepetitions)
	init_train_y = np.zeros(Nrepetitions)
	init_train_x = np.zeros((Nrepetitions,6))
	# final_regret = np.zeros(Nrepetitions)
	# safe_controllers = np.zeros(Nrepetitions)
	for k in range(Nrepetitions):
		aux_nums = np.arange(1,len(train_y_list[k])+1)
		ind_regret_finite = train_y_list[k] != init_unstable_value_for_EI_heur_high
		iter_num_first_stable_val_vec[k] = aux_nums[ind_regret_finite][0] # among all the infinities, take the first index
		init_train_y[k] = train_y_list[k][ind_regret_finite][0]
		init_train_x[k,:] = train_x_list[k][ind_regret_finite][0,:]
		# final_regret[k] = regret_simple_array_list[k][ind_regret_finite][-1]
		# safe_controllers[k] = np.sum(ind_regret_finite)

	ind_best = np.argmin(init_train_y)
	init_train_x_best = init_train_x[ind_best,:]

	print("init_train_x_best:",init_train_x_best)
	print("init_train_y_best:",init_train_y[ind_best])

	print("Iteration at which the first controller was found: {0:f} ({1:f})".format(np.mean(iter_num_first_stable_val_vec),np.std(iter_num_first_stable_val_vec)))

	path2model = "../../objectives/walker_env/2020-04-18_21-28-42-walker2d_s0" # Like 200 epochs, pars = np.array([[0.5 , 0.  , 0.5 , 0.5 , 0.5 , 0.75]]) kind of works in "real"
	# path2model = "./walker_env/2020-04-18_21-28-42-walker2d_s0" # Trained 2000 epochs
	max_steps_ep_replay = 1500
	num_episodes = 20 # Average the result over num_episodes episodes
	dim = 6
	obj_fun = WalkerObj(dim=dim,path2model=path2model,max_steps_ep_replay=max_steps_ep_replay,
											num_episodes=num_episodes,which_env2load="real",
											manual_constraint_thres=None,render=False)

	cons_fun = WalkerCons(obj_fun)

	pars = init_train_x_best
	x_in_torch = torch.from_numpy(pars.flatten()).to(device=device,dtype=dtype)
	ret = obj_fun(x_in_torch)
	print("Obj val:",ret)

	x_in_torch = torch.from_numpy(pars.flatten()).to(device=device,dtype=dtype)
	ret = cons_fun(x_in_torch)
	print("Obj cons:",ret)

	
	# Solution
	# ========
	# constraint_mean (plus thres):  4.107255841477128
	# constraint_std:  0.34610979957575416

	# Iteration at which the first controller was found: 21.875000 (15.351201)

	pdb.set_trace()


if __name__ == "__main__":

	main()