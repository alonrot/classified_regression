import numpy as np
from scipy.special import erf,erfcx
import numpy.linalg as la
import pdb
from classireg.utils.gaussian_tools import GaussianTools
import warnings
from classireg.utils.parsing import get_logger
import logging
logger = get_logger(__name__)
np.set_printoptions(linewidth=1000)

class ExpectationPropagation:
	'''
	Implementation of Expectation Propagation, based on Alg. 3.5. of the book
	by C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning.
	===================================================

	'''

	def __init__(self,prior_mean,prior_cov,Maxiter=10,required_precission=1e-8,verbosity=True):

		# Static attributes:
		self._Maxiter = Maxiter
		self._required_precission = required_precission
		self._verbosity = verbosity

		# Others:
		self.gt = GaussianTools()

		# Restart:
		self.dim = None
		self.restart(prior_mean,prior_cov)

	def __delete__(self, instance):
		del self.dim
		del self.prior_mean
		del self.prior_cov
		del self.Maxiter
		del self.verbosity

	def _error_checking(self,prior_mean,prior_cov):

		assert prior_mean is not None
		assert prior_mean.ndim == 1
		assert prior_mean.shape[0] > 0

		assert prior_cov is not None
		assert prior_cov.ndim == 2
		assert prior_cov.shape[0] > 0
		assert prior_cov.shape[1] > 0
		assert prior_cov.shape[0] == prior_cov.shape[1]

		assert prior_mean.shape[0] == prior_cov.shape[0]
		assert prior_mean.shape[0] == prior_cov.shape[1]

	def restart(self,prior_mean,prior_cov):

		self._error_checking(prior_mean,prior_cov)
		self.prior_mean = prior_mean
		self.prior_cov = prior_cov
		self.dim = self.prior_mean.shape[0]

		# Fix the prior covariance:
		self.prior_cov[:,:] = self.gt.fix_singular_matrix(self.prior_cov,what2fix="Fixing self.prior_cov while restarting EP ...")

	@property
	def verbosity(self):
		return self._verbosity

	@property
	def Maxiter(self):
		return self._Maxiter

	@property
	def required_precission(self):
		return self._required_precission

	def _update_cavity_parameters_for_factor_i(self,tau_til_i,nnu_til_i,mmu_i,var_i):

		if self.verbosity:
			print("")
			logger.info("@_update_cavity_parameters_for_factor_i:")
			logger.info("---------------------------------------")
			logger.info("Inputs:")
			logger.info("tau_til_i = {0:s}".format(str(tau_til_i)))
			logger.info("nnu_til_i = {0:s}".format(str(nnu_til_i)))
			logger.info("mmu_i = {0:s}".format(str(mmu_i)))
			logger.info("var_i = {0:s}".format(str(var_i)))

		tau_cav_i = 1./var_i - tau_til_i
		nnu_cav_i  = mmu_i/var_i - nnu_til_i

		# Catch error in case the cavity variance becaomes smaller than zero:
		if tau_cav_i < 0.0:
			self._NegativeVarianceError_in_update_cavity_parameters_for_factor_i(tau_cav_i,var_i,tau_til_i)

		if self.verbosity:
			print("")
			logger.info("Outputs:")
			logger.info("tau_cav_i = {0:s}".format(str(tau_cav_i)))
			logger.info("nnu_cav_i = {0:s}".format(str(nnu_cav_i)))

		return tau_cav_i, nnu_cav_i

	def _update_marginal_moments_for_factor_i(self,i,tau_cav_i,nnu_cav_i,marginal_moments_fun,*args):
		'''
		 These moments are problem-specific, and have to be recomputed for each likelihood
		'''
		if self.verbosity:
			print("")
			logger.info("@_update_marginal_moments_for_factor_i:")
			logger.info("---------------------------------------")
			print("")
			logger.info("Inputs:")
			logger.info("tau_cav_i = {0:s}".format(str(tau_cav_i)))
			logger.info("nnu_cav_i = {0:s}".format(str(nnu_cav_i)))

		var_cav_i = 1./tau_cav_i
		mmu_cav_i = var_cav_i * nnu_cav_i

		if self.verbosity:
			print("")
			logger.info("Inputs 2:")
			logger.info("var_cav_i = {0:s}".format(str(var_cav_i)))
			logger.info("mmu_cav_i = {0:s}".format(str(mmu_cav_i)))

		# Compute the moments
		var_KL_i, mmu_KL_i, logZ_KL_i = marginal_moments_fun(i,mmu_cav_i,var_cav_i,*args)

		if self.verbosity:
			print("")
			logger.info("Outputs:")
			logger.info("var_KL_i = {0:s}".format(str(var_KL_i)))
			logger.info("mmu_KL_i = {0:s}".format(str(mmu_KL_i)))
			logger.info("logZ_KL_i = {0:s}".format(str(logZ_KL_i)))

		return var_KL_i, mmu_KL_i, logZ_KL_i

	def _update_site_parameters_for_factor_i(self,var_KL_i,mmu_KL_i,logZ_KL_i,tau_cav_i,nnu_cav_i,tau_til_old_i):

		if self.verbosity:
			print("")
			logger.info("@_update_site_parameters_for_factor_i:")
			logger.info("---------------------------------------")
			print("")
			logger.info("Inputs:")
			logger.info("var_KL_i = {0:s}".format(str(var_KL_i)))
			logger.info("mmu_KL_i = {0:s}".format(str(mmu_KL_i)))
			logger.info("logZ_KL_i = {0:s}".format(str(logZ_KL_i)))
			logger.info("tau_cav_i = {0:s}".format(str(tau_cav_i)))
			logger.info("nnu_cav_i = {0:s}".format(str(nnu_cav_i)))
			logger.info("tau_til_old_i = {0:s}".format(str(tau_til_old_i)))

		# Catch error in case var_KL_i is 0.0:
		if var_KL_i < 0.0:
			raise ValueError("var_KL_i cannot be negative!")
		elif var_KL_i == 0.0:
			# warnings.warn("\nvar_KL_i = 0.0, adding numerical noise: var_KL_i += 1e-100")
			# var_KL_i += 1e-200
			# var_KL_i += 1e-100
			# pdb.set_trace()
			print("")
			logger.debug("var_KL_i = 0.0, Correcting by doing: 1./(tau_cav_i + tau_til_old_i)")
			if tau_cav_i + tau_til_old_i < 1e-300:
				var_KL_i = 1e300
				# pdb.set_trace()
			else:
				var_KL_i = 1./(tau_cav_i + tau_til_old_i)
			
			Delta_tau_til_i = 0.0
			# self._ZeroDivisionError_in__update_site_parameters_for_factor_i()
		else:
			Delta_tau_til_i = 1./var_KL_i - tau_cav_i - tau_til_old_i

		tau_til_new_i = tau_til_old_i + Delta_tau_til_i
		nnu_til_new_i = mmu_KL_i/var_KL_i - nnu_cav_i
		# Special case:
		# if tau_til_new_i < 0.0 and tau_til_new_i >= -1e-6:
		if tau_til_new_i < 0.0 and tau_til_new_i >= -1e-4:
			logger.debug("tau_til_new_i < 0.0, Correcting tau_til_new_i and nnu_til_new_i by setting them to zero")
			tau_til_new_i = 0.0
			nnu_til_new_i = 0.0
		# elif tau_til_new_i < -1e-6:
		elif tau_til_new_i < -1e-4:
			raise ValueError("tau_til_new_i = "+str(tau_til_new_i)+" is negative, and too big in absolute value...")
		# Compute the log-Zeroth order moment:
		# NOTE: logZ_til_i is not really used to compute the final logZ at the end of the EP loop
		# However, we compute it here numerically stable, in log-space:
		if tau_til_new_i == 0.0:
			# log_part1 = logZ_KL_i + 0.5*np.log(2*np.pi) + 0.5 * 700.0
			log_part1 = 700.0
			log_part2 = 0.0
		else:
			log_part1 = logZ_KL_i + 0.5*np.log(2*np.pi) + 0.5*np.log(1./tau_cav_i + 1./tau_til_new_i)
			log_part2 = 0.5*(nnu_cav_i/tau_cav_i-nnu_til_new_i/tau_til_new_i)**2/(1./tau_cav_i + 1./tau_til_new_i)
			# part1 = np.exp(logZ_KL_i)*np.sqrt(2*np.pi*(1./tau_cav_i + 1./tau_til_new_i))
			# part2 = np.exp( 0.5*(nnu_cav_i/tau_cav_i-nnu_til_new_i/tau_til_new_i)**2/(1./tau_cav_i + 1./tau_til_new_i) )
			# Z_til_i = part1*part2

		logZ_til_i = log_part1 + log_part2

		if self.verbosity:
			print("")
			logger.info("Outputs:")
			logger.info("Delta_tau_til_i = {0:s}".format(str(Delta_tau_til_i)))
			logger.info("tau_til_new_i = {0:s}".format(str(tau_til_new_i)))
			logger.info("nnu_til_new_i = {0:s}".format(str(nnu_til_new_i)))
			logger.info("log_part1 = {0:s}".format(str(log_part1)))
			logger.info("log_part2 = {0:s}".format(str(log_part2)))
			logger.info("logZ_til_i = {0:s}".format(str(logZ_til_i)))

		return tau_til_new_i, nnu_til_new_i, Delta_tau_til_i, logZ_til_i

	def _update_posterior_moments_at_iteration_i(self,i,Delta_tau_til_i,Sigma_old,nnu_til_vec):

		if self.verbosity:
			print("")
			logger.info("@_update_posterior_moments_at_iteration_i:")
			logger.info("---------------------------------------")
			print("")
			logger.info("Inputs:")
			logger.info("Delta_tau_til_i = {0:s}".format(str(Delta_tau_til_i)))
			logger.info("Sigma_old = {0:s}".format(str(Sigma_old)))
			logger.info("nnu_til_vec = {0:s}".format(str(nnu_til_vec)))

		si = Sigma_old[:,i]

		Sigma_old_ii = Sigma_old[i,i]

		Sigma_new = Sigma_old - Delta_tau_til_i/(1+Delta_tau_til_i*Sigma_old_ii) * np.outer(si,si)
		mmu_new = Sigma_new.dot(nnu_til_vec)

		if self.verbosity:
			print("")
			logger.info("Outputs:")
			logger.info("np.outer(si,si) = {0:s}".format(str(np.outer(si,si))))
			logger.info("Sigma_new = {0:s}".format(str(Sigma_new)))
			logger.info("mmu_new = {0:s}".format(str(mmu_new)))

		return Sigma_new, mmu_new

	def _initialize_EP(self):

		# mu 		= np.zeros(self.dim)
		mu 			= self.prior_mean
		Sigma 	= self.prior_cov

		# Tilde parameters:
		tau_til_vec = np.zeros(self.dim)
		nnu_til_vec = np.zeros(self.dim)
		logZ_til_vec = np.zeros(self.dim)

		# Cavity parameters:
		tau_cav_i_vec = np.zeros(self.dim)
		nnu_cav_i_vec = np.zeros(self.dim)
		logZ_KL_i_vec = np.zeros(self.dim)

		return mu, Sigma, tau_til_vec, nnu_til_vec, logZ_til_vec, tau_cav_i_vec, nnu_cav_i_vec, logZ_KL_i_vec

	def run_EP(self,marginal_moments_fun,*args_marginal_moments_fun):

		# Initialization:
		mu,Sigma,tau_til_vec,nnu_til_vec,logZ_til_vec,tau_cav_i_vec,nnu_cav_i_vec,logZ_KL_i_vec = self._initialize_EP()

		# Loop:
		n_iter = 0
		logZ_prev = 0.0
		achieved_prec = self.required_precission + 1.0
		while n_iter < self.Maxiter and achieved_prec > self.required_precission:

			for i in range(self.dim):

				if self.verbosity:
					print("\n")
					logger.info("============================================")
					logger.info("Iteration (n_iter): {0:d}".format(n_iter))
					logger.info("  Dimension (i): {0:d}".format(i))
					logger.info("============================================")

				# Compute cavity distribution:
				tau_cav_i,nnu_cav_i = self._update_cavity_parameters_for_factor_i(tau_til_vec[i],nnu_til_vec[i],mu[i],Sigma[i,i])
				tau_cav_i_vec[i] = tau_cav_i
				nnu_cav_i_vec[i] = nnu_cav_i

				# Compute the marginal moments:
				var_KL_i,mmu_KL_i,logZ_KL_i = self._update_marginal_moments_for_factor_i(i,tau_cav_i,nnu_cav_i,marginal_moments_fun,*args_marginal_moments_fun)
				logZ_KL_i_vec[i] = logZ_KL_i

				# Compute site parameters:
				tau_til_old_i = tau_til_vec[i]
				tau_til_new_i, nnu_til_new_i, Delta_tau_til_i, logZ_til_vec[i] = self._update_site_parameters_for_factor_i(var_KL_i,mmu_KL_i,logZ_KL_i,tau_cav_i,nnu_cav_i,tau_til_old_i)
				tau_til_vec[i] = tau_til_new_i
				nnu_til_vec[i] = nnu_til_new_i

				# # Update posterior moments: This step is ommited in the axisepmgp.m implementation
				# Note that, in Alg. 3.5 (p.58) from Rasmussen's book, line 10 updates the full Sigma, using Sigma_ii, which
				# is different from the var_i that is used in line 5.
				# The one in line 5 is taken before entering the for loop
				# Sigma, mu = self._update_posterior_moments_at_iteration_i(i,Delta_tau_til_i,Sigma,nnu_til_vec)

			# Update iteration counter:
			n_iter += 1

			# Last step: recompute the approximate posterior
			if np.any(tau_til_vec < 0):
				raise ValueError("@run_EP(): tau_til_vec = {0:s}".format(str(tau_til_vec)))

			S_til_sqrt = np.diag(np.sqrt(tau_til_vec))
			L = la.cholesky(np.eye(self.dim) + S_til_sqrt.dot(self.prior_cov).dot(S_til_sqrt))
			# Important note: la.cholesky() returns a lower triangular. Alg. 3.5 (p.58) from Rasmussen's book, line 14 assumes that
			# L is upper triangular, and that's why they transpose it. However, we don't need to transpose it.
			# In other words, la.cholesky() assumes B = L.dot(L.T), with B = I + S_til_sqrt*K*S_til_sqrt, while
			# chol() from matlab, and also Rasmussen's book assumes B = L'*L
			V = la.solve(L,S_til_sqrt.dot(self.prior_cov))
			Sigma = self.prior_cov - V.T.dot(V)
			
			# mu = Sigma.dot(nnu_til_vec)
			mu = Sigma.dot(la.solve(self.prior_cov,self.prior_mean) + nnu_til_vec)
			
			if self.verbosity:
				logger.info("tau_til_vec = {0:s}".format(str(tau_til_vec)))

			# logZ = self._compute_logZ_standard(logZ_til_vec,tau_til_vec,nnu_til_vec,mu,Sigma)
			logZ = self._compute_logZ_stable(logZ_KL_i_vec,tau_til_vec,nnu_til_vec,mu,Sigma,tau_cav_i_vec,nnu_cav_i_vec)

			# Update the achieved precision only if logZ is not insanely small:
			achieved_prec = np.fabs(logZ - logZ_prev)
			logZ_prev = logZ

		# if n_iter == self.Maxiter and self.verbosity:
			# print("")
			# print("n_iter = {0:d} / {1:d}".format(n_iter,self.Maxiter))
			# # logger.info("<<< WARNING >>>")
			# logger.info("Max. Number of iterations ({0:d})".format(self.Maxiter))
			# # logger.info("EP stopped because it reached the Max. Number of iterations ({0:d})".format(self.Maxiter))
			# # logger.info("but not because it reached the required precision.")
			# logger.info("Required precision: {0:20f}".format(self.required_precission))
			# print("Required precision: {0:20f}".format(self.required_precission))
			# logger.info("Achieved precision: {0:20f}".format(achieved_prec))
			# print("Achieved precision: {0:20f}".format(achieved_prec))
			# logger.info("               logZ: {0:20f}".format(logZ))

			# print("Max. Number of iterations ({0:d})".format(self.Maxiter))
			# # print("EP stopped because it reached the Max. Number of iterations ({0:d})".format(self.Maxiter))
			# # print("but not because it reached the required precision.")
			# print("Required precision: " + str(self.required_precission))
			# print("Achieved precision: " + str(achieved_prec))
			# print("               logZ: " + str(logZ))


		# else:
		# 	logger.info("Required precision: {0:20f}".format(self.required_precission))
		# 	print("Required precision: {0:20f}".format(self.required_precission))
		# 	logger.info("Achieved precision: {0:20f}".format(achieved_prec))
		# 	print("Achieved precision: {0:20f}".format(achieved_prec))


		# for i in range(self.dim):
		# 	logZ_vec[i] = self.compute_logZ_i_stable(i,logZ_KL_i_vec,tau_til_vec,nnu_til_vec,mu,Sigma,tau_cav_i_vec,nnu_cav_i_vec)
		# logger.info("logZ_vec\n==========")
		# logger.info(logZ_vec)
		# logger.info("Z_vec\n==========")
		# logger.info(np.exp(logZ_vec))
		# logger.info("Sum(logZ_vec)\n===========")
		# logger.info(np.sum(logZ_vec))
		# logger.info("Prod(Z_vec)\n===========")
		# logger.info(np.exp(np.sum(logZ_vec)))

		# Double check results:
		assert not np.isnan(logZ) and not np.isnan(logZ) and isinstance(logZ,float), "logZ must be a finite scalar"
		assert not np.any(np.isnan(Sigma)) and not np.any(np.isinf(Sigma)), "Sigma contains NaNs or Infs"
		assert not np.any(np.isnan(mu)) and not np.any(np.isinf(mu)), "mu contains NaNs or Infs"

		return Sigma, mu, logZ

	def _compute_logZ_standard(self,logZ_til_vec,tau_til_vec,nnu_til_vec,mu,Sigma):

		term1 = -0.5*( self.prior_mean.T.dot(la.solve(self.prior_cov,self.prior_mean)) + np.log(la.det(self.prior_cov)) )
		term2 = np.sum( logZ_til_vec -0.5*( nnu_til_vec**2/tau_til_vec - np.log(tau_til_vec) + np.ones(self.dim)*np.log(2*np.pi) ) )
		term3 = 0.5*( mu.T.dot(la.solve(Sigma,mu)) + np.log(la.det(Sigma)) )

		logZ =  term1 + term2 + term3

		return logZ

	def _compute_logZ_stable(self,logZ_KL_i_vec,tau_til_vec,nnu_til_vec,mu,Sigma,tau_cav_i_vec,nnu_cav_i_vec):
		'''
		Stable implementation from the EP paper
		'''
		Sigma[:,:] = self.gt.fix_singular_matrix(Sigma,what2fix="Fixing Sigma inside EP loop...")

		error_found = False
		with warnings.catch_warnings():
			warnings.filterwarnings('error')
			try:
				term1 = -0.5*( self.prior_mean.T.dot(la.solve(self.prior_cov,self.prior_mean)) + np.log(la.det(self.prior_cov)) )
				term2 = np.sum( logZ_KL_i_vec + 0.5*np.log(1+tau_til_vec/tau_cav_i_vec) + 0.5*(tau_til_vec*nnu_cav_i_vec**2/tau_cav_i_vec - 2*nnu_cav_i_vec*nnu_til_vec - nnu_til_vec**2)/(tau_cav_i_vec+tau_til_vec) )
				term3 = 0.5*( mu.T.dot(la.solve(Sigma,mu)) + np.log(la.det(Sigma)) )
			except Exception as inst:
				logger.info("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
				if inst.args[0] == "invalid value encountered in log":
					logger.info("la.det(self.prior_cov): {0:s}".format(str(la.det(self.prior_cov))))
					logger.info("la.det(Sigma): {0:s}".format(str(la.det(Sigma))))
					logger.info("tau_til_vec/tau_cav_i_vec: {0:s}".format(str(tau_til_vec/tau_cav_i_vec)))
					error_found = True

		if error_found:
			pdb.set_trace()

		logZ = term1 + term2 + term3

		return logZ

	def _ZeroDivisionError_in_update_site_parameters_for_factor_i(self):
		if self.verbosity == True:
			print("")
			logger.info("---------------------")
			logger.info("<<<< Error in EP >>>>")
			logger.info("---------------------")
			print("")
			logger.info("@ExpectationPropagation.update_site_parameters_for_factor_i()")
			logger.info("-------------------------------------------------------------")
			logger.info("The variance that minimizes the KL divergence became zero:")
			logger.info("var_KL_i = 0.0")
			logger.info("This is probably caused because the prior covariance is ill-conditioned")
			print("")
			logger.info("Prior covariance")
			logger.info("================")
			logger.info(str(self.prior_cov))
			cond_n = la.cond(self.prior_cov)
			eig_val = la.eigvals(self.prior_cov)
			print("")
			logger.info("Condition number")
			logger.info("================")
			logger.info(str(cond_n))
			print("")
			logger.info("Eigenvalues")
			logger.info("===========")
			logger.info(str(eig_val))
			print("")
			logger.info("Fix the prior covariance and try again...")
			logger.info("-------------------------------------------------------------")
			logger.info("-------------------------------------------------------------\n")
		raise ZeroDivisionError("var_KL_i = 0.0")

	def _NegativeVarianceError_in_update_cavity_parameters_for_factor_i(self,tau_cav_i,var_i,tau_til_i):
		if self.verbosity == True:
			print("")
			logger.info("---------------------")
			logger.info("<<<< Error in EP >>>>")
			logger.info("---------------------")
			print("")
			logger.info("@ExpectationPropagation.update_cavity_parameters_for_factor_i:")
			logger.info("--------------------------------------------------------------")
			logger.info("The variance of the cavity distribution became negative:")
			logger.info("tau_cav_i < 0.0")
			logger.info("tau_cav_i = 1./var_i - tau_til_i")
			print("")
			logger.info("Diagnose:")
			logger.info("tau_cav_i = ",tau_cav_i)
			logger.info("var_i = ",var_i)
			logger.info("tau_til_i = ",tau_til_i)
			# logger.info("\nPrior covariance")
			# logger.info("================")
			# logger.info(self.prior_cov)
			cond_n = la.cond(self.prior_cov)
			eig_val,eig_vec = la.eig(self.prior_cov)
			print("")
			logger.info("Condition number")
			logger.info("================")
			logger.info(cond_n)
			print("")
			logger.info("Eigenvalues")
			logger.info("===========")
			logger.info(eig_val)
			logger.info("-------------------------------------------------------------")
			logger.info("-------------------------------------------------------------\n")
		raise ValueError("tau_cav_i < 0.0")





