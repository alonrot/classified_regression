import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pdb
import numpy as np
np.set_printoptions(linewidth=400)
import time
import torch
from classireg.utils.parsing import display_banner, get_logger, move_logging_data
from classireg.utils.parse_data_collection import convert_from_cluster_data_to_single_file, obj_fun_list, generate_folder_at_path
import hydra
from omegaconf import DictConfig
import traceback
import time
logger = get_logger(__name__)
np.set_printoptions(linewidth=10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
list_algo = ["EIC","EI","EI_heur_high","EI_heur_low","EIClassi","EIC_standard"]

@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig) -> None:

    assert cfg.acqui in list_algo

    print(cfg.pretty())

    if cfg.acqui == "EIC" or cfg.acqui == "EIClassi" or cfg.acqui == "EIC_standard":
        from classireg.experiments.numerical_benchmarks.loop_BOC import run
    if cfg.acqui == "EI" or cfg.acqui == "EI_heur_high" or cfg.acqui == "EI_heur_low":
        from classireg.experiments.numerical_benchmarks.loop_BO  import run

    if cfg.run_type == "sequential": # For local computations

        # Create the folder only once and here:
        my_path = "./{0:s}/{1:s}_results".format(cfg.which_objective,cfg.acqui)
        path2data = generate_folder_at_path(my_path,create_folder=True) 

        for rep_nr in range(cfg.Ninit,cfg.Nend):
            display_banner(cfg.acqui,cfg.Nend,rep_nr+1)
            run(cfg, rep_nr, path2data)

        # Convert data to a single file:
        if cfg.which_objective != "simple1D" and cfg.which_objective != "quadruped8D":
            time.sleep(1.0) # Give some time for writing into the yaml files
            convert_from_cluster_data_to_single_file(which_obj=cfg.which_objective,which_acqui=cfg.acqui,Nrepetitions=cfg.Nend,path2data=path2data)

    elif cfg.run_type == "individual": # For parallel computations (e.g., in cluster)

        assert cfg.rep_nr is not None, "cfg.rep_nr is the iteration number and cannot be None, but cfg.rep_nr = {0,1,...,cfg.Nend-1}"
        assert cfg.rep_nr  < cfg.Nend, "required cfg.rep_nr < cfg.Nend, with cfg.rep_nr = {0,1,...,cfg.Nend-1}"

        display_banner(cfg.acqui,cfg.Nend,cfg.rep_nr+1)
        try:

            # Delay the execution by a tiny amount of time to avoid collisions:
            time.sleep(cfg.rep_nr/50.)

            # Create the folder only once and here:
            my_path = "./{0:s}/{1:s}_results".format(cfg.which_objective,cfg.acqui)
            path2data = generate_folder_at_path(my_path,create_folder=True,use_date=cfg.date_folder_name)

            run(cfg, cfg.rep_nr, path2data)

        except Exception as inst:
            logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
            traceback.print_exc()
            logger.info("{0:s} failed (!!) Moving logging data anyways ...".format(cfg.acqui))

        move_logging_data(path2data=path2data,which_acqui=cfg.acqui,which_obj=cfg.which_objective,rep_nr=cfg.rep_nr)

    else:
        raise ValueError("cfg.run_type = {sequential,individual}")


if __name__ == "__main__":
    main()


