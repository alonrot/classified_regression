import numpy as np
from scipy.special import erfcx
import warnings
import pdb

def marginal_moments_EP_unbounded_hyperrectangle(i,mmu_cav_i,var_cav_i,lim_lower,lim_upper):

	verbosity = False

	if verbosity == True:
		print("\n@marginal_moments_EP_ClassifiedRegression()")

	# Get integration limits of the hyperrectangle for the factor i:
	lim_lower_i = lim_lower[i]
	lim_upper_i = lim_upper[i]

	if np.any(var_cav_i < 0):
		raise ValueError("Some of the variances of the cavity distribution is negative!")

	sqrt_var_times_2 = np.sqrt(2*var_cav_i)
	sqrt_var_over_2pi = np.sqrt(var_cav_i/(2*np.pi))

	# flag_avoid_num_unstability = False
	if np.isinf(lim_lower_i):

		beta = (lim_upper_i-mmu_cav_i)/sqrt_var_times_2

		if beta > 26:
			logZhatOtherTail = np.log(0.5) + np.log(erfcx(beta)) - beta**2
			logZhat = np.log1p(-np.exp(logZhatOtherTail))
		else:
			logZhat = np.log(0.5) + np.log(erfcx(-beta)) - beta**2

		meanConst = -2./erfcx(-beta)
		varConst = -2./erfcx(-beta)*(lim_upper_i + mmu_cav_i)

		if verbosity == True:
			print("beta = ",beta)

	elif np.isinf(lim_upper_i):

		alpha = (lim_lower_i-mmu_cav_i)/sqrt_var_times_2

		if alpha < -26:
			logZhatOtherTail = np.log(0.5) + np.log(erfcx(-alpha)) - alpha**2
			logZhat = np.log1p(-np.exp(logZhatOtherTail))
		else:
			logZhat = np.log(0.5) + np.log(erfcx(alpha)) - alpha**2

		meanConst = 2./erfcx(alpha)
		varConst = 2./erfcx(alpha)*(lim_lower_i + mmu_cav_i)

		if verbosity == True:
			print("alpha = ",alpha)

	else:
		raise NotImplementedError("Cases beyond these two have yet to be implemented.\nHere, only unbounded hyperrectangles with one vertex are considered")

	if verbosity == True:
		print("logZhat = ",logZhat)

	# Normalization factor:
	logZ_KL_i = logZhat

  # Marginal mean:
	mmu_KL_i = mmu_cav_i + meanConst*sqrt_var_over_2pi

	# Marginal variance:
	var_KL_i = mmu_cav_i**2 + var_cav_i + varConst*sqrt_var_over_2pi - mmu_KL_i**2

	# The variance cannot be negative!
	if var_KL_i <= -1e-6:
		pdb.set_trace()
		# raise ValueError("var_KL_i must be positive")
		warnings.warn("\nvar_KL_i is not positive; var_KL_i <= {0:f}".format(-1e-6))
	elif var_KL_i > -1e-6 and var_KL_i < 0.0:
		warnings.warn("\nvar_KL_i was negatively small, so corrected it to 0.0")
		var_KL_i = 0.0

	if verbosity == True:
		print("\nDiagnose")
		print("========")
		print("logZ_KL_i = ",logZ_KL_i)
		print("mmu_KL_i = ",mmu_KL_i)
		print("mmu_cav_i = ",mmu_cav_i)
		print("meanConst = ",meanConst)
		print("var_KL_i = ",var_KL_i)
		print("var_cav_i = ",var_cav_i)
		print("varConst = ",varConst)
		print("sqrt_var_over_2pi = ",sqrt_var_over_2pi)

	# Double check results:
	assert not np.isnan(logZ_KL_i) and not np.isnan(logZ_KL_i) and isinstance(logZ_KL_i,float), "logZ_KL_i must be a finite scalar"
	assert not np.any(np.isnan(var_KL_i)) and not np.any(np.isinf(var_KL_i)), "var_KL_i must be a finite scalar"
	assert not np.any(np.isnan(mmu_KL_i)) and not np.any(np.isinf(mmu_KL_i)), "mmu_KL_i must be a finite scalar"

	return var_KL_i,mmu_KL_i,logZ_KL_i

	