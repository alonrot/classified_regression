import numpy as np
from torch import Tensor
from gpytorch.priors import GammaPrior, SmoothedBoxPrior, NormalPrior
import os
import yaml
import logging
import shutil
import pdb
list_algo = ["EIC","EI","EI_heur_high","EI_heur_low","EIClassi","EIC_standard"]
np.set_printoptions(linewidth=1000)

def print_list(list,name,usedim=False):
	print("\n{0:s} list\n=================".format(name))
	np.set_printoptions(precision=3)
	print(get_array_from_list(list,usedim=usedim))
	np.set_printoptions(precision=None)

def get_array_from_list(x_list):

	if isinstance(x_list,list) != True:
		return x_list

	# elif len(x_list) == 1 and x_list[0] is None:
	# 	raise ValueError("x_list cannot be an empty list")

	Nel = len(x_list)
	if Nel == 0:
		return None

	arr = np.asarray(x_list)
	if arr.ndim == 2:
		if arr.shape[1] == 1:
			arr = arr[:,0]

	return arr

def print_2Dtensor(tensor: Tensor ,name: str, dim: int) -> None:
	np_array = tensor.view((1,dim)).detach().cpu().numpy()
	np.set_printoptions(precision=3)
	print(np_array)
	np.set_printoptions(precision=None)

def extract_prior(cfg):

	if cfg.lengthscales.which == "box":
		lengthscale_prior = SmoothedBoxPrior(cfg.lengthscales.prior_box.lb, cfg.lengthscales.prior_box.ub, sigma=0.001)
	elif cfg.lengthscales.which == "gamma":
		lengthscale_prior = GammaPrior(concentration=cfg.lengthscales.prior_gamma.concentration, rate=cfg.lengthscales.prior_gamma.rate)
	elif cfg.lengthscales.which == "gaussian":
		lengthscale_prior = NormalPrior(loc=cfg.lengthscales.prior_gaussian.loc, scale=cfg.lengthscales.prior_gaussian.scale)
	else:
		lengthscale_prior = None
		print("Using no prior for the lengthscale")

	if cfg.outputscale.which == "box":
		outputscale_prior = SmoothedBoxPrior(cfg.outputscale.prior_box.lb, cfg.outputscale.prior_box.ub, sigma=0.001)
	elif cfg.outputscale.which == "gamma":
		outputscale_prior = GammaPrior(concentration=cfg.outputscale.prior_gamma.concentration, rate=cfg.outputscale.prior_gamma.rate)
	elif cfg.outputscale.which == "gaussian":
		outputscale_prior = NormalPrior(loc=cfg.outputscale.prior_gaussian.loc, scale=cfg.outputscale.prior_gaussian.scale)
	else:
		outputscale_prior = None
		print("Using no prior for the outputscale")

	return lengthscale_prior, outputscale_prior
	
def convert_lists2arrays(logvars):

	node2write = dict()
	for key, val in logvars.items():

		if "_list" in key:
			key_new = key.replace("_list","_array")
		else:
			key_new = key

		# import pdb; pdb.set_trace()
		node2write[key_new] = get_array_from_list(val)

	return node2write


def save_data(node2write: dict, which_obj: str, which_acqui: str, rep_nr: int) -> None:

	# Save data:
	path2obj = "./{0:s}".format(which_obj)
	if not os.path.exists(path2obj):
		print("Creating " + path2obj + " ...")
		os.makedirs(path2obj)

	path2results = path2obj + "/" + which_acqui + "_results"
	if not os.path.exists(path2results):
		print("Creating " + path2results + " ...")
		os.makedirs(path2results)

	path2save = path2results + "/cluster_data"
	if not os.path.exists(path2save):
		print("Creating " + path2save + " ...")
		os.makedirs(path2save)

	file2save = path2save + "/data_" + str(rep_nr) + ".yaml"

	print("\nSaving in {0:s} ...".format(file2save))

	with open(file2save, "w") as stream_write:
		yaml.dump(node2write, stream_write)


def move_logging_data(path2data: str, which_acqui: str, which_obj: str, rep_nr: int) -> None:

	# Create folders:
	# path2obj = "./{0:s}".format(which_obj)
	# path2results = path2obj + "/" + which_acqui + "_results"
	# path2save = path2results + "/" + exp_nr
	path2condor_logging = path2data + "/condor_logging"
	if not os.path.exists(path2condor_logging):
		print("Creating " + path2condor_logging + " ...")
		os.makedirs(path2condor_logging,exist_ok=False)

	# Move logging files:
	ext_list = ["err","out","log"]
	for extension in ext_list:
		file_name = "condor_{0:s}_{1:d}.{2:s}".format(which_acqui,rep_nr,extension) # This name must coincide with the structure of the {.err,.out,.log} files in ./config/cluster/launch_XX.sub files
		file_and_path = "./{0:s}/{1:s}".format(which_obj,file_name)
		assert os.path.exists(file_and_path), "The path {0:s} does not exist".format(file_and_path) # make sure the files exist (they are automatically generated when running the jobs on cluster)
		try:
			new_location = shutil.move(src=file_and_path,dst="{0:s}/{1:s}".format(path2condor_logging,file_name))
		except:
			print("File {0:s} could not be moved to {1:s}/{2:s} ...".format(file_name,path2condor_logging,file_name))
		else:
			print("File successfully moved to {0:s}".format(new_location))

def display_banner(which_algo,Nrep,rep_nr):

	assert which_algo in list_algo

	if which_algo == "EIC":
		algo_name = "Expected improvement with crash constraints (EIC**2) - Modeling the constraint with GPCR"
	if which_algo == "EIC_standard":
		algo_name = "Expected improvement with constraints (EIC) - Modeling the constraint with a standard GP"
	if which_algo == "EIClassi":
		algo_name = "Expected improvement with constraints (EIC) - Modeling the constraint with a GP classifier"
	if which_algo == "EI":
		algo_name = "Expected improvement (adaptive heuristic cost for failing controllers)"
	if which_algo == "EI_heur_high":
		algo_name = "Expected improvement (pre-defined high heuristic cost)"
	if which_algo == "EI_heur_low":
		algo_name = "Expected improvement (pre-defined low heuristic cost)"
	banner_name = " <<<<<<<<<<<<<<<<<<< {0:s} >>>>>>>>>>>>>>>>>>> ".format(algo_name)
	line_banner = "="*len(banner_name)

	print(line_banner)
	print(banner_name)
	print(line_banner)
	print(" * Running a total of {0:d} repetition(s)".format(Nrep))
	print(" * Repetition {0:d} / {1:d}".format(rep_nr,Nrep))
	print("")


def get_logger(name,level=logging.INFO):

	logger = logging.getLogger(name)
	ch = logging.StreamHandler()
	ch.setLevel(level)
	formatter = logging.Formatter('[%(name)s] %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	logger.setLevel(level)

	return logger


