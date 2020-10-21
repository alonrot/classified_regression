import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from classireg.utils.parsing import get_logger
logger = get_logger(__name__)
np.set_printoptions(linewidth=10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
INF = float("Inf")
import yaml
import pdb
from botorch.utils.sampling import draw_sobol_samples

class QuadrupedObj():

	def __init__(self,dim):
		"""
		
		x_in domain: [-5.0 , +5.0]
		
		"""
		self.dim = dim
		self.cons_value = None

	def collect_float_positive(self,which_fun):

		# Get cost value:
		correct_num = False
		while not correct_num:

			aux = input(" * Enter the {0:s} value (expected a positive float): ".format(which_fun))

			is_float = True
			try:
				aux = float(aux)
			except:
				is_float = False
			else:
				if np.isinf(aux) or np.isnan(aux):
					is_float = False

			if not is_float:
				logger.info("Please, enter a float number. Your entry was: {0:s}".format(str(aux)))
			elif aux <= 0.0:
				logger.info("Please, enter a positive float number. Your entry was: {0:s}".format(str(aux)))
			else:
				correct_num = True

		return aux

	def collect_value_manual_input(self):

		values_are_correct = False
		while not values_are_correct:

			# Sort out failure/success:
			aux = 999
			while aux not in ["suc","fail"]:
				aux = input(" * Was the experiment a failure or a success ? Type 'suc' or 'fail' : ")
			is_stable = aux == "suc"
			
			val_cost = val_constraint = INF
			if is_stable:
				val_cost = self.collect_float_positive(which_fun="f(x) (cost)")
				val_constraint = self.collect_float_positive(which_fun="g(x) (constraint)")

			logger.info("Here is a summary of the values:")
			logger.info("    Label:            {0:s}".format("Success!" if is_stable == True else "Failure (!)"))
			logger.info("    Cost value:       {0:5f}".format(val_cost))
			logger.info("    constraint value: {0:5f}".format(val_constraint))
			logger.info("Are you ok to continue? If not, you'll be asked to enetr all numbers once more.")

			while aux not in ["0","1"]:
				aux = input(" * Please type [1] Yes | [0] No: ")

			if aux == "1":
				values_are_correct = True

		return is_stable, val_cost, val_constraint

	def _parsing(self,x_in):
		"""
		
		"""
		assert x_in.dim() == 1, "Squeeze the vector right before..."

		par = torch.zeros(x_in.shape[0])

		# ------ 8D -----------------------------

		# # kp_joint_min:
		# par[0] = 1.0 + (2.0-1.0)*x_in[0] # hip
		# par[1] = 1.0 + (2.0-1.0)*x_in[1] # knee

		# # kp_joint_max:
		# par[2] = 2.0 + (10.0-2.0)*x_in[2] # hip
		# par[3] = 2.0 + (10.0-2.0)*x_in[3] # knee

		# # kd_joint_min:
		# par[4] = 0.03 + (0.08-0.03)*x_in[4] # hip
		# par[5] = 0.03 + (0.08-0.03)*x_in[5] # knee

		# # kd_joint_max:
		# par[6] = 0.08 + (0.14-0.08)*x_in[6] # hip
		# par[7] = 0.08 + (0.14-0.08)*x_in[7] # knee


		# ------ 4D ----------------------------- (23 and 27 Jul 2020)

		# kp_joint_max:
		par[0] = 2.0 + (7.0-2.0)*x_in[0] # hip
		par[1] = 2.0 + (7.0-2.0)*x_in[1] # knee

		# # kd_joint_min (0-35 iters) -> 23 Jul 2020
		# par[2] = 0.03 + (0.08-0.03)*x_in[2] # hip
		# par[3] = 0.03 + (0.08-0.03)*x_in[3] # knee

		# kd_joint_min (0-50 iters) -> 27 jul 2020
		par[2] = 0.02 + (0.14-0.02)*x_in[2] # hip
		par[3] = 0.02 + (0.14-0.02)*x_in[3] # knee


		# ------ 5D ----------------------------- (29 Jul 2020)

		# # kp_joint_max:
		# par[0] = 1.5 + (5.0-1.5)*x_in[0] # hip
		# par[1] = 1.5 + (5.0-1.5)*x_in[1] # knee

		# # kd_joint_min:
		# par[2] = 0.02 + (0.14-0.02)*x_in[2] # hip
		# par[3] = 0.02 + (0.14-0.02)*x_in[3] # knee

		# # Foot location:
		# par[4] = 0.15 + (0.20-0.15)*x_in[4] # Foot location

		return par


	def evaluate(self,x_in,with_noise=False):

		str_banner = " <<<< Collecting new evaluation >>>> "
		logger.info("="*len(str_banner))
		logger.info(str_banner)
		logger.info("="*len(str_banner))

		try:
			x_in = self.error_checking_x_in(x_in)
		except:
			logger.info("Saturating...")
			x_in[x_in > 1.0] = 1.0
			x_in[x_in < 0.0] = 0.0

		x_in = x_in.squeeze()

		# Domain transformation:
		par = self._parsing(x_in)

		# logger.info("")
		# logger.info("Summary of gains")
		# logger.info("================")
		# logger.info(" ++++ KNEE ++++ ")
		# logger.info("----------------")
		# logger.info(" * P-GAIN:")
		# logger.info("    Initial: {0:2.5f} | x[0]: {1:2.5f}".format(par[0],x_in[0]))
		# logger.info("    Final:   {0:2.5f} | x[1]: {1:2.5f}".format(par[1],x_in[1]))
		# logger.info(" * D-GAIN:")
		# logger.info("    Initial: {0:2.5f} | x[2]: {1:2.5f}".format(par[2],x_in[2]))
		# logger.info("    Final:   {0:2.5f} | x[3]: {1:2.5f}".format(par[3],x_in[3]))
		# logger.info("")
		# logger.info(" ++++ HIP ++++ ")
		# logger.info("----------------")
		# logger.info(" * P-GAIN:")
		# logger.info("    Initial: {0:2.5f} | x[4]: {1:2.5f}".format(par[4],x_in[4]))
		# logger.info("    Final:   {0:2.5f} | x[5]: {1:2.5f}".format(par[5],x_in[5]))
		# logger.info(" * D-GAIN:")
		# logger.info("    Initial: {0:2.5f} | x[6]: {1:2.5f}".format(par[6],x_in[6]))
		# logger.info("    Final:   {0:2.5f} | x[7]: {1:2.5f}".format(par[7],x_in[7]))
		# logger.info("")

		logger.info("")
		logger.info("Summary of gains")
		logger.info("================")
		# logger.info("kp_joint_min: [{0:2.4f} , {1:2.4f}]".format(par[0].item(),par[1].item()))
		logger.info("kp_joint_max: [{0:2.4f} , {1:2.4f}]".format(par[0].item(),par[1].item()))
		logger.info("")
		logger.info("kd_joint_min: [{0:2.4f} , {1:2.4f}]".format(par[2].item(),par[3].item()))
		# logger.info("kd_joint_max: [{0:2.4f} , {1:2.4f}]".format(par[6].item(),par[7].item()))
		logger.info("")
		logger.info("Foot location: {0:2.4f}".format(par[4].item()))

		# Request cost value:
		is_stable, val_cost, val_constraint = self.collect_value_manual_input()

		# # Re-scaling if necessary:
		if val_cost != float("Inf") and val_constraint != float("Inf"):
			val_cost 				= -10.0 * val_cost + 8.0
			val_constraint	= (val_constraint-60)/10.0 # Optimistic mean
			# val_constraint	= (val_constraint - 120.0)/10.0 # Pessimistic mean
			logger.info("    [re-scaled] Cost value:       {0:2.4f}".format(val_cost))
			logger.info("    [re-scaled] constraint value: {0:2.4f}".format(val_constraint))

		# Place -1.0 labels and INF to unstable values:
		l_out = (+1.0)*is_stable + (-1.0)*(not is_stable)

		# Assign constraint value (the constraint WalkerCons must be called immediately after):
		self.cons_value = torch.tensor([[val_constraint, l_out]],device=device,dtype=dtype)
		return torch.tensor([val_cost],device=device,dtype=dtype)

	def error_checking_x_in(self,x_in):

		x_in = x_in.view(-1,self.dim)
		assert x_in.dim() == 2, "x_in does not have the proper size"
		assert not torch.any(torch.isnan(x_in)), "x_in contains nans"
		assert not torch.any(torch.isinf(x_in)), "x_in contains Infs"
		assert torch.all(x_in <= 1.0), "The input parameters must be inside the unit hypercube"
		assert torch.all(x_in >= 0.0), "The input parameters must be inside the unit hypercube"
		assert x_in.shape[0] == 1, "We shall pass only one initial datapoint"
		return x_in

	def __call__(self,x_in,with_noise=False):
		return self.evaluate(x_in,with_noise=with_noise)

	@staticmethod
	def true_minimum():
		x_gm = torch.tensor([[0.5]*4],device=device,dtype=dtype)
		f_gm = 0.0
		return x_gm, f_gm

class QuadrupedCons():
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

	# dim = 8
	# dim = 4
	dim = 5
	obj_fun = QuadrupedObj(dim=dim)
	cons_fun = QuadrupedCons(obj_fun)

	train_x = draw_sobol_samples(bounds=torch.tensor([[0.]*dim,[1.]*dim]),n=1,q=1).squeeze(1) # Get only unstable evaluations


	# # 4D points:
	# # train_x = torch.tensor([[3.3298e-01, 1.0000e+00, 4.5455e-01, 1.3440e-02]])
	# train_x = torch.tensor([[0.6223, 0.9955, 0.4433, 0.5836]]) # Max height 2020 Jul 27, after 13:27 train_y_obj_min: tensor(0.1400)
	# # train_x = torch.tensor([[0.7979372, 0.9929017, 0.71484804, 0.03631881]]) # Max height 2020 Jul 27, after 13:27 train_y_obj_min: tensor(0.1600)




	val_cost = obj_fun(train_x)
	val_constraint = cons_fun(train_x)
	is_stable = val_constraint[0,1] == +1

	logger.info("Entered values:")
	logger.info("    Label:            {0:s}".format("Success!" if is_stable == True else "Failure (!)"))
	logger.info("    Cost value:       {0:5f}".format(val_cost.item()))
	logger.info("    constraint value: {0:5f}".format(val_constraint[0,0]))
	logger.info("Are you ok to continue? If not, you'll be asked to enetr all numbers once more.")



