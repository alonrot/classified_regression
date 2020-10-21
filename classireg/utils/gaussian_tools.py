import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.integrate import quad,dblquad, simps
import pdb # debug
import warnings

class GaussianTools():
	'''

	Construct class GaussianTools for univariate Gaussians
	======================================================
	'''

	def product_gaussian_densities_different_dimensionality(self,mu1,Sigma1,mu12,Sigma12,get_scaling_fac=False,verbosity=False):
		'''
		mu1: [N1 x 1]
		Sigma1: [N1 x N1]
		mu12: [(N1+N2) x 1] This has to be zero-mean
		Sigma1: [(N1+N2) x (N1+N2)]
		'''

		if mu1 is None or Sigma1 is None:
			if get_scaling_fac is True:
				return Sigma12,mu12,None
			else:
				return Sigma12,mu12

		# Ensure mu1 is a 1-D vector, and that dimensions agree:
		mu1 	= self.err_check_input_mean_cov(mu1,Sigma1)
		mu12 	= self.err_check_input_mean_cov(mu12,Sigma12)

		if verbosity == True:
			print("\n-----------------------------------------------------------")
			print("Product of Gaussian densities with different dimensionality")
			print("-----------------------------------------------------------")

		# Dimensions:
		N1 	= len(mu1)
		N12 = len(mu12)

		flag_same_dim = False
		if N12 < N1:
			raise ValueError("The large dimension must be N12 >= N1")
		elif N12 == N1:
			flag_same_dim = True

		N2 = N12-N1

		# Variances:
		Sigma1_inv = la.inv(Sigma1)
		if flag_same_dim == True:
			Sigma1_inv_ext = Sigma1_inv
		else:
			Sigma1_inv_ext = np.block([ [Sigma1_inv, np.zeros((N1,N2))],
																[np.zeros((N2,N1)), np.zeros((N2,N2))] ])

		# Quadratic form:
		C = la.inv(Sigma12)+Sigma1_inv_ext
		if flag_same_dim == True:
			b = -2*mu1.T.dot(Sigma1_inv).T
		else:
			b = np.block([ [-2*mu1.T.dot(Sigma1_inv) , np.zeros(N2)] ]).T

		a = mu1.T.dot(Sigma1_inv).dot(mu1)

		# Transformation:
		D = la.inv(C)
		m = -0.5*D.dot(b)
		v = a - 0.25*b.T.dot(D).dot(b)

		# New density is N([x1;x2],m,D)*beta
		if get_scaling_fac:
			beta = np.sqrt( np.exp(-v)*la.det(D)/(2*np.pi)**(N1)/la.det(Sigma1)/la.det(Sigma12) )
			beta = np.asscalar(beta)

			if beta is None or np.isscalar(beta) == False or np.isnan(beta) == True:
				# pdb.set_trace()
				raise ValueError("beta should be a positive scalar!")

		# Double-check dimensionality:
		m = self.err_check_input_mean_cov(m,D)

		if verbosity == True:
			print("\nResulting mean")
			print("==============")
			print(m)
			print("\nResulting covariance")
			print("====================")
			print(D)
			print("\nScaling factor (beta)")
			print("=====================")
			print(beta)

		if get_scaling_fac == True:
			return D,m,beta
		else:
			return D,m

	def err_check_input_mean_cov(self,mean,cov):

		# If mean is a 2D array, transform it into a vector:
		if len(mean.shape) > 1:
			mean_out = np.squeeze(mean)
		else:
			mean_out = mean

		if cov.ndim != 2:
			print("cov:",cov)
		if cov.shape[0] != cov.shape[1]:
			raise ValueError("@GaussianTools.err_check_input_mean_cov(): cov has to be square")

		if len(mean) != cov.shape[0]:
			raise ValueError("@GaussianTools.err_check_input_mean_cov(): Length of mean is not the same as dimensions of cov")

		return mean_out

	def fix_singular_matrix(self,singular_mat,verbosity=False,what2fix=None,val_min_deter=1e-200,val_max_cond=1e9):

		assert singular_mat.ndim == 2
		assert singular_mat.shape[0] == singular_mat.shape[1]

		# Corner case:
		cond_num = la.cond(singular_mat)
		deter = la.det(singular_mat)
		# val_min_deter = 1e-320
		# val_min_deter = 1e-200
		# val_max_cond = 1e12
		# val_max_cond = 1e9
		
		# Check positive definiteness:
		chol_ok = True
		try: la.cholesky(singular_mat)
		except Exception as inst:
			if verbosity == True:
				print(type(inst),inst.args)
			chol_ok = False

		# Check log(det)
		log_det_ok = True
		try:
			with warnings.catch_warnings():
				warnings.filterwarnings('error')
				np.log(deter)
		except Exception as inst:
			if verbosity == True:
				print(type(inst),inst.args)
			log_det_ok = False

		if cond_num <= val_max_cond and deter > val_min_deter and chol_ok == True and log_det_ok == True:
			return singular_mat
		else:
			pass
			# print("@GaussianTools.fix_singular_matrix(): singular_mat needs to be fixed")
			# if what2fix is not None: print("what2fix:",what2fix)

		# Get the order of magnitude of the largest eigenvalue in singular_mat, assuming all eigenvalues are positive:
		eigs_real = np.real(la.eigvals(singular_mat))
		largest_eig = np.amax(eigs_real)
		if largest_eig < 1e-310:
			max_ord = np.floor(np.log10(1e-310))
		else:
			max_ord = np.ceil(np.log10(largest_eig))

		# print("largest_eig: ",largest_eig)
		# print("max_ord: ",max_ord)

		# Get the order of magnitude of the smallest eigenvalue in singular_mat, assuming all eigenvalues are positive:
		smallest_eig = np.amin(eigs_real)
		if smallest_eig < 1e-310:
			min_ord = np.floor(np.log10(1e-310))
		else:
			min_ord = np.floor(np.log10(np.abs(smallest_eig)))
		
		# Initial factor:
		fac_init = min_ord*2.

		if verbosity == True:
			print("\n[VERBOSITY]: @GaussianTools.fix_singular_matrix(): singular_mat needs to be fixed")
			print("cond_num:",cond_num)
			print("min_ord:",min_ord)
			print("max_ord:",max_ord)
			print("chol_ok:",chol_ok)
			print("log_det_ok:",log_det_ok)
			print("Before update:")
			print("==============")
			print("fac_init:",fac_init)
			print("order cond_num:",np.floor(np.log10(cond_num)))
			print("deter:",deter)
			print("eig:",la.eigvals(singular_mat))

		# Fix the matrix:
		Id = np.eye(singular_mat.shape[0])
		singular_mat_new = singular_mat
		c = 0
		singular = True
		fac = 10**(fac_init)
		while singular == True and fac_init + c < max_ord:

			# New factor:
			fac = 10**(fac_init+c)
			singular_mat_new[:,:] = singular_mat + fac*Id

			# Look for errors:
			try:
				with warnings.catch_warnings():
					warnings.filterwarnings('error')
					la.cholesky(singular_mat_new)
					assert la.det(singular_mat_new) > val_min_deter
					np.log(la.det(singular_mat_new))
					assert la.cond(singular_mat_new) <= val_max_cond
			except Exception as inst:
				if verbosity == True:
					print(type(inst),inst.args)
				c += 1
			else:
				singular = False

		if verbosity == True:
			print("After update:")
			print("=============")
			print("fac:",fac)
			print("order cond_num:",np.floor(np.log10(la.cond(singular_mat_new))))
			print("deter:",la.det(singular_mat_new))
			print("eig:",la.eigvals(singular_mat_new))

		if singular == True:
			# pdb.set_trace()
			# raise ValueError("Matrix could not be fixed. Something is really wrong here...")
			# warnings.warn("Matrix could not be fixed. Something is really wrong here...")
			print("Matrix could not be fixed. Something is really wrong here...") # Highest permission

		return singular_mat_new

