from ObjectiveFunction import ObjectiveFunction
import ctypes
import numpy as np
import numpy.random as npr
import warnings
import pdb

# Transform into numpy array:
def ctype_double_ptr_from_np(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

class CostFromApollo(ObjectiveFunction):

	def __init__(self,dim,iden="call-apollo",alternate=False):

		ObjectiveFunction.__init__(self,dim=dim,noise_std=0.0,implicit_thres=0.0,iden=iden) # in python3: super().__init__(GPmodel.dim) || in general ObjectiveFunction.__init__(GPmodel.dim)

		# Handle library object:
		try:
			self.hl = ctypes.CDLL('libuserES_pubsub_lqr.so')
			# self.hl = ctypes.CDLL('/is/am/amarcovalle/popspace/workspace/devel/lib/libuserES_pubsub_lqr.so')
		except Exception as inst:
			print type(inst),inst.args
			raise ValueError("The library libuserES_pubsub_lqr.so could not be found. Try:\nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/is/am/amarcovalle/popspace/workspace/devel/lib")
		else:
			self.my_print("Library to communicate with Apollo loaded correctly!")

		# Define the function interface of the exposed functions:
		self.hl.cost_function_init.argtypes = []
		self.hl.cost_function_init.restype = ctypes.c_voidp

		self.hl.cost_function_call.argtypes = [	ctypes.POINTER(ctypes.c_double),ctypes.c_int,
																						ctypes.POINTER(ctypes.c_double),ctypes.c_int,
																						ctypes.POINTER(ctypes.c_bool),
																						ctypes.POINTER(ctypes.c_bool)]
		self.hl.cost_function_call.restype = ctypes.c_voidp

		# Call the function as named in c++:
		# The constructor reads from a yaml file many parameters.
		# Once passed this line, modifying such parameters will have no effect here
		self.hl.cost_function_init()

		# We assume couple evaluations, so the function will be called once, and will store both, cost value and constraint value
		# We use the labels that are also obatined to determine constraint satisfaction

		self.prev_what2read = "g(x)"
		self.stored_constraint_value = 0.0
		self.stored_constraint_label = True
		self.alternate = alternate

	def check_what2read(self,what2read):
		if what2read != "f(x)" and what2read != "g(x)":
			raise ValueError("what2read must be:\n'f(x) or 'g(x)'")

	def evaluate(self,x_in,with_noise=True,return_constraint=False,what2read="f(x)"):
		'''
		Exceptionally, override also evaluate, which defined in the parent ObjectiveFunction

		The idea is to communicate with the robot (i.e., call to self.hl.cost_function_call())
		only once, and store the acquired constraint value for the next time self.evaluate() is called
		For security, objective and constraint can only be called alternating with each other,
		never two repeated calls to f(x) or g(x).
		'''

		self.check_what2read(what2read)

		# The same
		if what2read == "f(x)":

			if self.alternate == True and self.prev_what2read != "g(x)":
				pdb.set_trace()
				raise ValueError("Evaluating f(x)... This should not happen!!!")

			x_in = self.error_checking_x_in(x_in)
			assert x_in.shape[0] == 1 # Ensure a single point
			x_in = x_in.flatten() # Make it flat
			assert x_in.ndim == 1

			# Output values:
			# out_values(0) = this->cost_value;
			# out_values(1) = this->max_endeff_pos_value;
			obj_and_cons_values = np.zeros(2)

			# Label for the cost: Inside the endeffector acceleration? (absorbed by f(x))
			# This value will be passed by reference
			# label_acc = ctypes.c_bool(True)
			label_cost = ctypes.c_bool(True)
			# label_cost = True

			# Label for the constraint: Inside the endeffector box? (independent g(x))
			# This value will be passed by reference
			# label_ss = ctypes.c_bool(True)
			label_cons = ctypes.c_bool(True)
			# label_cons = True

			# Call the function as named in c++:
			# void cost_function_call(double * pars, int Dim, double * out_values, int Dim_out)
			self.hl.cost_function_call(ctype_double_ptr_from_np(x_in),len(x_in), # in
																ctype_double_ptr_from_np(obj_and_cons_values),len(obj_and_cons_values), # out
																label_cost,label_cons) # out

			# Extract values and labels:
			f_obs = obj_and_cons_values[0]
			f_is_stable = label_cost.value
			g_obs = obj_and_cons_values[1]
			g_is_stable = label_cons.value

			try:

				assert isinstance(f_obs,float)
				assert isinstance(g_obs,float)
				assert isinstance(f_is_stable,bool)
				assert isinstance(g_is_stable,bool)

			except Exception:
				pdb.set_trace()

			# Cases:
			if self.alternate == True:

				# Verbosity:
				if f_is_stable == True and g_is_stable == True:
					self.my_print("<f(x) STABLE>   |   <g(x) STABLE>")
				elif f_is_stable == False:
					self.my_print("<f(x) UNSTABLE>   |   <g(x) STABLE>")
				elif g_is_stable == False:
					self.my_print("<f(x) STABLE>   |   <g(x) UNSTABLE>")
				else:
					self.my_print("<f(x) UNSTABLE>   |   <g(x) UNSTABLE>")
	
				# Fill in output:
				if f_is_stable == True:
					y_out = f_obs
					l_out = -1 # STA
				else:
					y_out = None
					l_out = +1 # UNS
	
			else:

				# Verbosity:
				if f_is_stable == True and g_is_stable == True:
					self.my_print("<f(x) STABLE>")
				else:
					self.my_print("<f(x) UNSTABLE>")

				# Fill in output:
				if f_is_stable == True and g_is_stable == True:
					y_out = f_obs
					l_out = -1 # STA
				else:
					y_out = None
					l_out = +1 # UNS


			self.my_print(" f(x) = "+str(f_obs))
			self.my_print(" g(x) = "+str(g_obs))

			# Construct hybrid observation:
			yl_out = np.array([y_out,l_out])

			# Store the constraint value and the label for later:
			self.stored_constraint_value = g_obs
			self.stored_constraint_label = g_is_stable

			# Security:
			self.prev_what2read = "f(x)"

		elif what2read == "g(x)":

			if self.alternate == True and self.prev_what2read != "f(x)":
				pdb.set_trace()
				raise ValueError("Evaluating g(x)... This should not happen!!!")

			if self.stored_constraint_label == True:
				y_out = self.stored_constraint_value
				l_out = -1 # STA
			else:
				y_out = None
				l_out = +1 # UNS

			yl_out = np.array([y_out,l_out])

			# Security:
			self.prev_what2read = "g(x)"

		return yl_out

	def evaluate_obj_fun(self):
	# def get_value(self,pars,out_values):

		# Call the function as named in c++:
		# void cost_function_call(double * pars, int Dim, double * out_values, int Dim_out)
		self.hl.cost_function_call(ctype_double_ptr_from_np(pars),len(pars),
															ctype_double_ptr_from_np(out_values),len(out_values))

		return out_values

	def evaluate_cons_fun(self,x,with_noise=False):
		raise NotImplementedError("The constraint g(x) must be implemented by child classes. It must be such that g(x) < 0.0 implies constraint satisfaction")

	def plot(self,axes=None,block=False,Ndiv=41):
		warnings.warn("No plotting is possible")
		pass

if __name__ == "__main__":

	cfa = CostFromApollo()

	# yl_eval = function_obj.evaluate(xnext,with_noise=False,return_constraint=True)

	my_pars = npr.uniform(low=0.0,high=1.0,size=(1))
	print "my_pars:",my_pars

	out_values = np.zeros(2)

	ccf.get_value(my_pars,out_values)
	print "out_values: ",out_values