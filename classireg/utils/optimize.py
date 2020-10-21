import nlopt
import numpy as np
import torch
import pdb
from classireg.utils.parsing import get_logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
logger = get_logger(__name__)

class ConstrainedOptimizationNonLinearConstraints():

	def __init__(self,dim,fun_obj,fun_cons,tol_cons=1e-5,tol_x=1e-3,Neval_max_local_optis=200,what2optimize_str=""):


		# Safe Constrained problem:
		self.hdl_opt_nonlin_cons = nlopt.opt(nlopt.LN_COBYLA,dim)
		self.hdl_opt_nonlin_cons.set_max_objective(self.fun_obj_wrap)
		self.hdl_opt_nonlin_cons.add_inequality_constraint(self.fun_cons_wrap,tol_cons) # This tolerance is absolute. nlopt assumes g(x) <= 0 as constraint satisfaction
		self.hdl_opt_nonlin_cons.set_lower_bounds(0.0)
		self.hdl_opt_nonlin_cons.set_upper_bounds(1.0)
		# self.hdl_opt_nonlin_cons.set_ftol_rel(tol_acqui*1e-1)
		self.hdl_opt_nonlin_cons.set_xtol_rel(tol_x*1e-1)
		self.hdl_opt_nonlin_cons.set_maxeval(Neval_max_local_optis)

		self.tol_cons_in_opti = 10*tol_cons

		self.fun_obj = fun_obj
		self.fun_cons = fun_cons
		self.dim = dim

		if len(what2optimize_str) == 0:
			self.what2optimize_str = "Optimization with nlopt"
		else:
			self.what2optimize_str = "Optimizing {0:s} with nlopt".format(what2optimize_str)

	# Interface between pytorch and numpy:
	def fun_obj_wrap(self,x_in, grad_in=np.array([])):
		if not torch.is_tensor(x_in):
			x_in = torch.from_numpy(x_in).to(device=device,dtype=dtype).view((1,1,self.dim))
		val = self.fun_obj(x_in)
		assert len(val) == 1
		return val.item()

	# Interface between pytorch and numpy:
	def fun_cons_wrap(self,x_in, grad_in=np.array([])):
		if not torch.is_tensor(x_in):
			x_in = torch.from_numpy(x_in).to(device=device,dtype=dtype).view((1,1,self.dim))
		val = self.fun_cons(x_in)
		assert len(val) == 1
		return val.item()

	def run_constrained_minimization(self, x_restarts: torch.Tensor):

		x_restarts = x_restarts.detach().cpu().numpy()
		Nrestarts_local = x_restarts.shape[0]
		x_next_vec = np.zeros((Nrestarts_local,self.dim))
		alpha_val_xnext = np.zeros(Nrestarts_local)
		ind_cons_violated = np.ones(Nrestarts_local,dtype=np.bool) # All initialized to True
		logger.info("{0:s} | Nrestarts_local = {1:d}".format(self.what2optimize_str,Nrestarts_local))
		for i in range(Nrestarts_local):

			if (i+1) % 2 == 0:
				logger.info("Optimizer restarted {0:d} / {1:d} times".format(i+1,Nrestarts_local))

			try:
				x_next_vec[i,:] = self.hdl_opt_nonlin_cons.optimize(x_restarts[i])
			except Exception as inst:
				logger.info(type(inst),inst.args)
				alpha_val_xnext[i] = np.nan # Assign NaN to the cases where the optimization fails
			else:

				# Store the optimization value:
				alpha_val_xnext[i] = self.hdl_opt_nonlin_cons.last_optimum_value()

				# Sanity check: If a solution is found, alpha_val_xnext[i] should never be inf, nan or None
				if np.isnan(alpha_val_xnext[i]) or np.isinf(alpha_val_xnext[i]) or alpha_val_xnext[i] is None:
					logger.info("Sanity check failed: The optimizer returns a successful state, but alpha=NaN")

				cons_val = self.fun_cons_wrap(x_next_vec[i])
				if cons_val > self.tol_cons_in_opti:
					logger.info("x_next_vec["+str(i)+"] = "+str(x_next_vec[i,:])+" wasn't a good point because it exceeds the constraint tolerance")
					logger.info("Required cons_val <= "+str(self.tol_cons_in_opti)+", cons_val = "+str(cons_val))
				else:
					ind_cons_violated[i] = False

		# Check for points that violate the constraint:
		if np.all(np.isnan(alpha_val_xnext)):
			logger.info("Something went really wrong here...")
			# pdb.set_trace()
			raise ValueError("Fatal error: Optimizer returned NaN in all cases because (i) the acquisition function is corrupted, OR (ii) no feasible solution was found. Abort (!)")
		elif np.all(ind_cons_violated):
			logger.info("No feasible solution was found. Think about increasing the number of random restarts...")
			logger.info("Return the unconstrained minimum...")
			ind_next = np.nanargmin(alpha_val_xnext) # We call nanargmax in case some of the points were exceptions
			x_next = np.atleast_2d(x_next_vec[ind_next])
			alpha_next = alpha_val_xnext[ind_next]
		else:
			logger.info("The constrainted problem was succesfully solved!")
			ind_next = np.argmin(alpha_val_xnext[~ind_cons_violated]) # We call nanargmax in case some of the points were exceptions
			x_next = np.atleast_2d(x_next_vec[~ind_cons_violated][ind_next])
			alpha_next = alpha_val_xnext[~ind_cons_violated][ind_next]

		# Get tensors:
		x_next = torch.from_numpy(x_next).to(device=device,dtype=dtype)
		alpha_next = torch.Tensor([alpha_next]).to(device=device,dtype=dtype)

		logger.info("{0:s} | Done!".format(self.what2optimize_str))
		return x_next, alpha_next


class OptimizationNonLinear():

	def __init__(self,dim,fun_obj,algo_str='LN_COBYLA',tol_x=1e-3,Neval_max_local_optis=200,bounds=None,minimize=True,what2optimize_str=""):


		# Safe Constrained problem:
		algo_name = eval('nlopt.{0:s}'.format(algo_str))
		self.hdl_opt_nonlin = nlopt.opt(algo_name,dim)
		self.minimize = minimize
		if self.minimize:
			self.hdl_opt_nonlin.set_min_objective(self.fun_obj_wrap)
		else:
			self.hdl_opt_nonlin.set_max_objective(self.fun_obj_wrap)
		self.bounds = bounds
		if self.bounds is not None:
			self.hdl_opt_nonlin.set_lower_bounds(self.bounds[0])
			self.hdl_opt_nonlin.set_upper_bounds(self.bounds[1])
		else:
			self.hdl_opt_nonlin.set_lower_bounds(0.0)
			self.hdl_opt_nonlin.set_upper_bounds(1.0)
		# self.hdl_opt_nonlin.set_ftol_rel(tol_acqui*1e-1)
		self.hdl_opt_nonlin.set_xtol_rel(tol_x*1e-1)
		self.hdl_opt_nonlin.set_maxeval(Neval_max_local_optis)

		self.fun_obj = fun_obj
		self.dim = dim

		if len(what2optimize_str) == 0:
			self.what2optimize_str = "Optimization with nlopt"
		else:
			self.what2optimize_str = "Optimizing {0:s} with nlopt".format(what2optimize_str)


	# Interface between pytorch and numpy:
	def fun_obj_wrap(self,x_in, grad_in=np.array([])):
		if not torch.is_tensor(x_in):
			x_in = torch.from_numpy(x_in).to(device=device,dtype=dtype)

		# # Ensure a two dimensional X:
		# if x_in.dim() == 1:
		# 	x_in = x_in.view(-1,self.dim)

		val = self.fun_obj(x_in)

		if torch.is_tensor(val):
			assert val.dim() <= 1, "val must be a 1D tensor with a single element"
			if val.dim() == 1:
				if not val.shape[0] == 1:
					pdb.set_trace()
					
			val = val.item()

		return val

	def run_optimization(self, x_restarts: torch.Tensor):

		x_restarts = x_restarts.detach().cpu().numpy()
		Nrestarts_local = x_restarts.shape[0]
		x_next_vec = np.zeros((Nrestarts_local,self.dim))
		alpha_val_xnext = np.zeros(Nrestarts_local)
		logger.info("{0:s} | Nrestarts_local = {1:d}".format(self.what2optimize_str,Nrestarts_local))
		for i in range(Nrestarts_local):

			if (i+1) % 2 == 0:
				logger.info("Optimizer restarted {0:d} / {1:d} times".format(i+1,Nrestarts_local))

			try:
				x_next_vec[i,:] = self.hdl_opt_nonlin.optimize(x_restarts[i])
			# except nlopt.RoundoffLimited as inst:
			# 	logger.info(str(type(inst))+str(inst.args))
			except Exception as inst:
				logger.info(str(type(inst))+str(inst.args))
				alpha_val_xnext[i] = np.nan # Assign NaN to the cases where the optimization fails
				# pdb.set_trace()
				# raise
			else:

				# Store the optimization value:
				alpha_val_xnext[i] = self.hdl_opt_nonlin.last_optimum_value()

				# Sanity check: If a solution is found, alpha_val_xnext[i] should never be inf, nan or None
				if np.isnan(alpha_val_xnext[i]) or np.isinf(alpha_val_xnext[i]) or alpha_val_xnext[i] is None:
					logger.info("Sanity check failed: The optimizer returns a successful state, but alpha=NaN")

		# Check for points that violate the constraint:
		if np.all(np.isnan(alpha_val_xnext)):
			logger.info("Something went really wrong here. If this happens, there's no way out...")
			# pdb.set_trace()
			raise ValueError("Fatal error: Optimizer returned NaN in all cases because (i) the acquisition function is corrupted, OR (ii) no feasible solution was found. Abort (!)")
		else:
			logger.info("The optimization problem was succesfully solved!")
			if self.minimize:
				ind_next = np.nanargmin(alpha_val_xnext) # We call nanargmax in case some of the points were exceptions
			else:
				ind_next = np.nanargmax(alpha_val_xnext) # We call nanargmax in case some of the points were exceptions
			x_next = np.atleast_2d(x_next_vec[ind_next])
			alpha_next = alpha_val_xnext[ind_next]
			# logger.info("x_next_vec: {0:s}".format(str(x_next_vec)))

		# Get tensors:
		x_next = torch.from_numpy(x_next).to(device=device,dtype=dtype)
		alpha_next = torch.tensor([alpha_next]).to(device=device,dtype=dtype)

		bounds_low = torch.tensor(self.bounds[0],device=device,dtype=dtype)
		bounds_high = torch.tensor(self.bounds[1],device=device,dtype=dtype)

		# Check that the result is inside the bounds:
		try:
			assert not torch.any(x_next.flatten() < bounds_low) and not torch.any(x_next.flatten() > bounds_high), "x_next out of boundaries"
		except:
			logger.info("x_next: {0:s}".format(str(x_next)))
			logger.info("x_next[0]: {0:s}".format(str(x_next[0])))
			logger.info("self.bounds[0]: {0:s}".format(str(bounds_low)))
			logger.info("self.bounds[1]: {0:s}".format(str(bounds_high)))
			
			# Clip:
			logger.info("Clipping...")
			ind_below = x_next.flatten() < bounds_low
			x_next[0,ind_below] = bounds_low[ind_below]
			ind_above = x_next.flatten() > bounds_high
			x_next[0,ind_above] = bounds_high[ind_above]

		logger.info("{0:s} | Done!".format(self.what2optimize_str))
		return x_next, alpha_next

