import numpy as np
import pdb
from gym import envs
import gym
import matplotlib.pyplot as plt
import time
import joblib
import os.path as osp
import torch
from classireg.objectives.walker_env import Walker2dEnv_modified
from gym.wrappers.time_limit import TimeLimit
INF = float("Inf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
import yaml

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds

def obj_wrap(x_in):
	x_in_torch = torch.from_numpy(x_in.flatten()).to(device=device,dtype=dtype)
	ret = obj_fun(x_in_torch)
	return ret.cpu().numpy()

def cons_wrap(x_in):
	x_in_torch = torch.from_numpy(x_in.flatten()).to(device=device,dtype=dtype)
	ret = cons_fun(x_in_torch)
	return ret.cpu().numpy()

def one_eval(obj_fun,cons_fun,pars):

	obj_val	= obj_wrap(pars)
	cons_val = cons_wrap(pars)

	print("obj_val: {0:2.4f}".format(obj_val.item()))
	print("cons_val: {0:s}".format(str(cons_val)))


def search_minimum(obj_fun,cons_fun,Neval,dim):

	# Select method:
	# which_method = "trust-constr"
	which_method = "SLSQP"
	# which_method = "COBYLA" #  Method COBYLA cannot handle bounds.

	if which_method == "COBYLA" or which_method == "SLSQP":

		nonlinear_constraint = {'type': 'ineq', 'fun' : lambda x: -cons_wrap(x)} # COBYLA, SLSQP
		options={'ftol': 1e-9, 'disp': True, 'maxiter': Neval}

	elif which_method == "trust-constr":

		nonlinear_constraint = NonlinearConstraint(cons_wrap, -np.inf, 0.0) # 'trust-constr'
		options={'verbose': 2, 'maxiter': Neval}

	else:
		raise ValueError("which_method ... ")

	Nrestarts = 10
	assert Neval % Nrestarts == 0, "Neval must be a multiple of {0:d} ...".format(Nrestarts)
	bounds = Bounds([0]*dim, [1.0]*dim)
	for ii in range(Nrestarts):

		print("    Nr. restart: {0:d} / {1:d}. Nr. max iterations per restart: {2:f}".format(ii+1,Nrestarts,Neval / Nrestarts))
		x0 = np.random.uniform(0.0,1.0,dim)
		print("    x0: {0:s}".format(str(x0)))

		res = minimize(obj_wrap, x0.flatten(), method=which_method, constraints=[nonlinear_constraint], options=options, bounds=bounds)

		print("\nResult:")
		for key,val in res.items():
			if key in ["success","message","fun","constr_violation","x"]:
				print("    {0:s}: {1:s}".format(key,str(val)))

	file2save = "./walker_env/walker_data_{0:d}D_{1:s}.yaml".format(dim,which_method)
	print("Saving in {0:s}".format(file2save))
	stream_write = open(file2save, "w")
	yaml.dump(res,stream_write)
	stream_write.close()

def brute_force_minimum(obj_fun,cons_fun,Neval,dim,name2save):

	Ndiv_dim = round(Neval**(1/dim))
	x_vec = [np.linspace(0, 1, Ndiv_dim)]*dim
	out = np.meshgrid(*x_vec)
	x_in_vec = np.array([out_xi.reshape(-1) for out_xi in out]).transpose()
	Neval = x_in_vec.shape[0]
	print("Neval:",Neval)
	print("Ndiv_dim:",Ndiv_dim)

	print("Running brute force ...")
	obj_val_vec = np.zeros(Neval)
	cons_val_vec = np.zeros((Neval,2))
	for k in range(Neval):

		if (k+1) % 100 == 0:
			print("iter: {0:d} / {1:d}".format(k+1,Neval))
			print("Nr. stable points: {0:d}".format(np.sum(cons_val_vec[:,1] == +1)))

		obj_val_vec[k] 	= obj_wrap(x_in_vec[k,:].reshape(-1,dim))
		cons_val_vec[k,:] = cons_wrap(x_in_vec[k,:].reshape(-1,dim))

	stable_points = cons_val_vec[:,1] == +1
	stable_points_ind = np.arange(1,len(stable_points)+1)[stable_points]
	Nstable = np.sum(stable_points)
	x_in_stable = x_in_vec[stable_points,:]
	print("obj_val_vec: ",obj_val_vec)
	print("cons_val_vec: ",cons_val_vec)
	print("x_in_stable:",x_in_stable)
	print("Nstable:",Nstable)
	print("stable_points:",stable_points)
	print("stable_points_ind:",stable_points_ind)

	node2write = dict()
	node2write['obj_val_vec'] = obj_val_vec
	node2write['cons_val_vec'] = cons_val_vec
	node2write['x_in_stable'] = x_in_stable
	node2write['Nstable'] = Nstable
	node2write['stable_points'] = stable_points
	node2write['stable_points_ind'] = stable_points_ind

	file2save = "./walker_env/walker_data_{0:d}D.yaml".format(dim)
	print("Saving in {0:s}".format(file2save))
	stream_write = open(file2save, "w")
	yaml.dump(node2write,stream_write)
	stream_write.close()

def brute_force_analysis():

	# filename = "walker_data_6D_brute_force_multipliying.yaml" # 26 / 15625 points are safe ~0.2%
	filename = "walker_data_6D_brute_force_multipliying.yaml" # 26 / 15625 points are safe ~0.2%
	file2load = "./walker_env/{0:s}".format(filename)
	print("Loading {0:s} ...".format(file2load))
	stream_load = open(file2load, "r")
	my_node = yaml.load(stream_load,Loader=yaml.UnsafeLoader)
	stream_load.close()

	print("Stable points values:")
	print(my_node["obj_val_vec"][my_node["stable_points"]])
	print("Stable points:")
	print(my_node["x_in_stable"])

	pdb.set_trace()

def get_env_and_policy(pars,max_episode_steps,fpath,which_env2load,itr=""):

	fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
	# print('Loading from %s.'%fname)

	# Load mode:
	model = torch.load(fname)

	# Change parameters:
	if pars is not None:

		assert pars.dim() in [1,2], "pars must be a 1- or 2-dimensional array"
		if pars.dim() == 2:
			assert pars.shape[0] == 1
			pars = pars.flatten()

		state_dict = model.state_dict()

		# Parametrize bias vector of the last layer; D = 1:
		# state_dict["pi.mu_net.4.bias"][:] *= pars.item()

		# Parametrize bias vector of the last layer; D = 3:
		# state_dict["pi.mu_net.4.bias"][0:3] = pars[0,:]

		# Parametrize bias vector of the last layer; D = 6:
		# state_dict["pi.mu_net.4.bias"] = pars[0,:] # tensor([ 0.1439,  0.1482,  0.1315,  0.1717,  0.0204, -0.0386])

		# Affect all the weights:
		# state_dict.keys(): ['pi.log_std', 'pi.mu_net.0.weight', 'pi.mu_net.0.bias', 'pi.mu_net.2.weight', 'pi.mu_net.2.bias', 'pi.mu_net.4.weight', 'pi.mu_net.4.bias', 'v.v_net.0.weight', 'v.v_net.0.bias', 'v.v_net.2.weight', 'v.v_net.2.bias', 'v.v_net.4.weight', 'v.v_net.4.bias']
		# pdb.set_trace()
		ii = 0
		for key in state_dict.keys():
			if "pi.mu_net" in key:
				state_dict[key] = state_dict[key] * pars[ii].item()
				ii += 1
		assert ii == pars.shape[0], "The dimensionality fo the parameter space in BO must coincide with that of the parametrized NN"

		model.load_state_dict(state_dict)

	# make function for producing an action given a single state
	def get_action(x):
		with torch.no_grad():
			x = torch.as_tensor(x, dtype=torch.float32)
			action = model.act(x)
		return action

	# which_env2load == "simu": Loads gym.envs.mujoco.__init__.py -> Walker2d-v2, which loads internally gym.envs.mujoco.walker2d.py -> Walker2dEnv(), 
	# wrapped with gym.wrappers.time_limit.py -> TimeLimit.step()
	#
	# which_env2load == "real": Loads classireg.objectives.walker_env.__init__.py -> Walker2dmodified-v1, which loads internally
	# classireg.objectives.walker_env.walker2d_mod.Walker2dEnv_modified()
	if which_env2load == "simu":
		state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
		env = state['env']
	elif which_env2load == "real":
		env = gym.make('Walker2dmodified-v1')
	else:
		raise ValueError("which_env2load = {simu,real}")

	# Modify the internal steps limit:
	# The episode will finished if _max_episode_steps is met.
	# env.step() calls first TimeLimit.step(), which internally calls Walker2dEnv.step()
	# See /Users/alonrot/.anaconda3_install/envs/gymwalker/lib/python3.6/site-packages/gym/wrappers/time_limit.py class TimeLimit()
	env._max_episode_steps = max_episode_steps

	return env, get_action


def run_policy(env, get_action, max_steps_ep_replay, num_episodes, render=False, manual_constraint_thres=None):

	assert env is not None, \
		"Environment not found!\n\n It looks like the environment wasn't saved, " + \
		"and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
		"page on Experiment Outputs for how to handle this situation."

	stable_vec = np.zeros(num_episodes,dtype=bool)
	cost_vec = np.zeros(num_episodes)
	constraint_vec = np.zeros(num_episodes)
	angle_torso = np.zeros(max_steps_ep_replay)

	o, r, done, ep_ret, ep_len, n, cc = env.reset(), 0, False, 0, 0, 0, 0
	while n < num_episodes:
		if render:
			env.render()
			time.sleep(1e-3)

		# angle_torso[cc] = env.sim.data.qpos[2] # posafter, height, ang = self.sim.data.qpos[0:3]
		# cc += 1

		# Step:
		a = get_action(o)
		o, r, done, _ = env.step(a) # done=True *only* when the robot falls down, see classireg.objectives.walker_env.walker2d_mod.py  -> Walker2dEnv_modified()

		# Manually specify a constraint threshold:
		if manual_constraint_thres is not None:
			done = np.abs(env.sim.data.qpos[2]) >= manual_constraint_thres # Prematurely finished if constraint violated
			# done = np.abs(env.sim.data.qpos[2]) >= 6.653581/10 # Learned threshold using EIC 

		ep_ret += r
		ep_len += 1
		angle_torso[cc] = env.sim.data.qpos[2] # posafter, height, ang = self.sim.data.qpos[0:3]
		cc += 1

		# Diagnose:
		if ep_len == max_steps_ep_replay or done:
			# print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))

			# Analize stability:
			if ep_len == max_steps_ep_replay: # If experiment completed -> stable
				is_stable_epi = True
			else: # if experiment not completed is because the angle limit was reached, hence unstable
				is_stable_epi = False
			# print("is_stable_epi: {0:s}".format(str(is_stable_epi)))
			stable_vec[n] = is_stable_epi

			# Objective cost: flip sign and scale:
			cost_vec[n] = -ep_ret / 1000.0

			# Constraint value: scale
			# constraint_vec[n] = np.max(np.abs(angle_torso)).item() * 10.0 - 10.0 # CHANGE ALSO walker.yaml | We subtract the upper bound on the constraint, in order to have a pessimistic mean
			constraint_vec[n] = np.max(np.abs(angle_torso)).item() * 10.0 # CHANGE ALSO walker.yaml | We don't subtract anything, and we get an optimistic mean
			angle_torso[:] = 0.0
			cc = 0

			# Restart:
			o, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
			n += 1
	
	# Final cost and stability:
	is_stable = np.sum(stable_vec) >= round(.9*num_episodes)
	if is_stable:
		cost_val = np.mean(cost_vec[stable_vec])
		# constraint_val = np.mean(constraint_vec[stable_vec])
		constraint_val = np.amax(constraint_vec[stable_vec])
	else:
		cost_val = constraint_val = INF

	# print("constraint_vec:",constraint_vec)
	# print("constraint_vec[stable_vec]:",constraint_vec[stable_vec])

	# print("constraint_mean (plus thres): ",np.mean(constraint_vec[stable_vec]) + 10.0)
	# print("constraint_std: ",np.std(constraint_vec[stable_vec]))

	# print("Stable {0:d} / {1:d}".format(np.sum(stable_vec),num_episodes))
	# print("stable_vec:",stable_vec)
	# print("cost_val: {0:2.4f}".format(cost_val))
	# print("constraint_val: {0:2.4f}".format(constraint_val))
	# print("is_stable: {0:s}".format(str(is_stable)))

	env.close()

	return cost_val, constraint_val, is_stable
	# return np.mean(cost_vec), (np.mean(constraint_vec)-10.0)*is_stable + (not is_stable)*1.0, is_stable # DEBUG
	# return np.mean(cost_vec), np.mean(constraint_vec)-8.0, is_stable # DEBUG

class WalkerObj():

	def __init__(self,dim,path2model,max_steps_ep_replay,num_episodes,which_env2load,manual_constraint_thres,render=False):
		"""
		
		x_in domain: [-5.0 , +5.0]
		
		"""
		self.dim = dim
		self.path2model = path2model
		self.max_steps_ep_replay = max_steps_ep_replay
		self.num_episodes = num_episodes
		self.cons_value = None
		self.render = render
		self.which_env2load = which_env2load
		self.manual_constraint_thres = manual_constraint_thres

	def evaluate(self,x_in,with_noise=False):

		try:
			x_in = self.error_checking_x_in(x_in)
		except:
			# print("Saturating...")
			x_in[x_in > 1.0] = 1.0
			x_in[x_in < 0.0] = 0.0

		# Re-scale to the domain:
		# pars = -5.0 + 10.0*x_in 			# bias last layer parametrization
		# pars = -1.0 + 2.0*x_in 				# bias last layer parametrization
		# pars = 10**(-1.0 + 2.0*x_in) 	# multiplicative in all layers
		pars = 10**(-0.5 + 1.0*x_in) 		# multiplicative in all layers

		# Run mujoco experiments:
		env, get_action = get_env_and_policy(	pars=pars,max_episode_steps=self.max_steps_ep_replay+500,
																					fpath=self.path2model,which_env2load=self.which_env2load,itr="")
		cost_val, constraint_val, is_stable = run_policy(env, get_action, max_steps_ep_replay=self.max_steps_ep_replay, 
																											num_episodes=self.num_episodes, render=self.render,
																											manual_constraint_thres=self.manual_constraint_thres)

		# Place -1.0 labels and INF to unstable values:
		l_out = (+1.0)*is_stable + (-1.0)*(not is_stable)
		y_out = constraint_val

		# Assign constraint value (the constraint WalkerCons must be called immediately after):
		self.cons_value = torch.tensor([[y_out, l_out]],device=device,dtype=dtype)
		return torch.tensor([cost_val],device=device,dtype=dtype)

		# # DEBUG:
		# self.cons_value = torch.tensor([constraint_val])
		# return torch.tensor([cost_val])

	def error_checking_x_in(self,x_in):

		x_in = x_in.view(-1,self.dim)
		assert x_in.dim() == 2, "x_in does not have the proper size"
		assert not torch.any(torch.isnan(x_in)), "x_in contains nans"
		assert not torch.any(torch.isinf(x_in)), "x_in contains Infs"
		assert torch.all(x_in <= 1.0), "The input parameters must be inside the unit hypercube"
		assert torch.all(x_in >= 0.0), "The input parameters must be inside the unit hypercube"
		return x_in

	def __call__(self,x_in,with_noise=False):
		return self.evaluate(x_in,with_noise=with_noise)

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[0.5]],device=device,dtype=dtype)
		f_gm = -4.0
		return x_gm, f_gm

class WalkerCons():
	def __init__(self,obj_inst):
		self.obj_inst = obj_inst
	def evaluate(self,x_in,with_noise=False):
		# In some cases, the constraint needs to be evaluated before the objective class has been called:
		if self.obj_inst.cons_value is None:
			self.obj_inst(x_in)
		return self.obj_inst.cons_value
	def __call__(self,x_in,with_noise=False):
		return self.evaluate(x_in,with_noise=with_noise)


if __name__ == "__main__":

	path2model = "./walker_env/2020-04-18_21-28-42-walker2d_s0" # Like 200 epochs, pars = np.array([[0.5 , 0.  , 0.5 , 0.5 , 0.5 , 0.75]]) kind of works in "real"
	# path2model = "./walker_env/2020-04-18_21-28-42-walker2d_s0" # Trained 2000 epochs
	max_steps_ep_replay = 1500
	num_episodes = 100 # Average the result over num_episodes episodes
	dim = 6
	obj_fun = WalkerObj(dim=dim,path2model=path2model,max_steps_ep_replay=max_steps_ep_replay,
											num_episodes=num_episodes,which_env2load="real",
											manual_constraint_thres=None,render=False)

	cons_fun = WalkerCons(obj_fun)

	# pars = np.array([[0.5]*dim]) # No change
	# pars = np.array([[0.5 , 0.  , 0.5 , 0.5 , 0.5 , 0.75]]) # Found with brute force, x_in \in 10**[-0.5,0.5], multiplicative
	# pars = np.array([[0.5 , 0.25 , 1. ,  0.75 ,  0.25 , 0.5]]) # Found with brute force, x_in \in 10**[-0.5,0.5], multiplicative (works 9/10)

	# Search initial unstable point, close to stable one: (see objectives/walker_env/walker_data_6D_brute_force_multipliying_trying_many.txt)
	# pars = np.array([[0.5  ,0.25, 0.5,  0.25, 0.5,  0.75]]) # Found with brute force, x_in \in 10**[-0.5,0.5], multiplicative | Stable 94 / 100
	# pars = np.array([[0.4869, 0.2519, 0.5054, 0.2278, 0.5026, 0.7397]]) # Stable 96 / 100
	# pars = np.array([[0.45715266 0.2105419  0.47219464 0.26393669 0.48742414 0.73103866]]) # Stable 93 / 100
	# pars = np.array([[0.48757593 0.04239111 0.57527158 0.11504335 0.51823669 0.7683684 ]]) # Stable 91 / 100
	# pars = np.array([[0.4212881  0.39294725 0.35245743 0.398108   0.48078322 0.6231427 ]]) # Stable 0 / 100
	# pars: [[0.27630052 0.21404426 0.52598395 0.36198066 0.53788427 0.6246932 ]] # Stable 50 / 100 | 0.15
	# pars: [[0.19493661 0.43998336 0.34638286 0.4183597  0.53057703 0.44843165]] # Stable 0 / 100 | 0.15
	# pars: [[0.47180474 0.25449639 0.56117447 0.32954418 0.53956743 0.80124233]] # Stable 96 / 100 | 0.10
	# pars: [[0.50819633 0.         0.57875323 0.22328409 0.5534009  0.72277186]] # Stable 88 / 100 | 0.12
	# pars = np.array([[0.67110407, 0.24342352, 0.71659071, 0.37363523, 0.52991535, 0.49756885]]) # Stable 65 / 100 | 0.12
	# pars = np.array([[0.5  ,0.25, 0.5,  0.25, 0.5,  0.75]]) + 0.12*np.random.randn(6)

	# Best controller found for which_env2load="real" using EIC:
	pars = np.array([[0.5630794 , 0.141994  , 0.61658806, 0.04447119, 0.4978954 , 0.9797242 ]]) # y_best: -6.291757 # Stable 93 / 100 # obj_val: -2.8226 # cons_val: [[-4.6548157  1.       ]]
																																															# With angle threshold 6.653581/10, we have: Stable 92 / 100 # obj_val: -2.8237 # cons_val: [[-5.2688985  1.       ]]
	
	pars[pars < 0] = 0
	pars[pars > 1] = 1
	print("pars:",pars)
	
	one_eval(obj_fun,cons_fun,pars)

	# search_minimum(obj_fun,cons_fun,Neval=2000,dim=dim)

	# brute_force_minimum(obj_fun,cons_fun,Neval=3**6,dim=dim,name2save=path2model)

	# brute_force_analysis()







