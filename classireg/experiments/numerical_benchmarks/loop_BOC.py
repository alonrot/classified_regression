import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pdb
import numpy as np
import torch
from classireg.experiments.numerical_benchmarks.loop_utils import initialize_logging_variables,append_logging_variables,get_initial_evaluations,get_objective_functions
from classireg.utils.plotting_collection import plotting_tool_cons
from classireg.acquisitions import ExpectedImprovementWithConstraints, ExpectedImprovementWithConstraintsClassi
from classireg.utils.parsing import convert_lists2arrays, save_data, get_logger
from classireg.utils.parse_data_collection import generate_folder_at_path
from omegaconf import DictConfig
from botorch.models import ModelListGP
from classireg.models.gpmodel import GPmodel
from classireg.models.gpcr_model import GPCRmodel
from classireg.models.gpclassi_model import GPClassifier
import yaml
import traceback
logger = get_logger(__name__)
np.set_printoptions(linewidth=10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32



def run(cfg: DictConfig, rep_nr: int) -> None:

    # Random seed for numpy and torch:
    np.random.seed(rep_nr)
    torch.manual_seed(rep_nr)

    # Load true function and initial evaluations:
    function_obj, function_cons, dim, x_min, f_min = get_objective_functions(which_objective=cfg.which_objective)
    
    logvars = initialize_logging_variables()

    if cfg.safety_mechanisms.load_from_file.use:

        nr_exp = cfg.safety_mechanisms.load_from_file.nr_exp
        path2data = "./{0:s}/{1:s}_results/{2:s}".format(cfg.which_objective,cfg.acqui,nr_exp)
        try:
            with open("{0:s}/data_0.yaml".format(path2data), "r") as stream:
                my_node = yaml.load(stream,Loader=yaml.UnsafeLoader)
        except Exception as inst:
            logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
            raise ValueError("Data corrupted or non-existent!!!")
        else:
            logger.info("We have lodaded existing data from {0:s}".format(path2data))
            logger.info("A quick inspection reveals {0:d} existing datapoint(s) ...".format(len(my_node["regret_simple_array"])))

        if cfg.safety_mechanisms.load_from_file.modify:
            logger.info("Here, we have the opportunity of modifying some data, if needed...")
            pdb.set_trace()

        # pdb.set_trace()
        # Get stored values:
        if my_node["GPs"][0]["train_inputs"] is not None:
            train_x_obj = torch.from_numpy(my_node["GPs"][0]["train_inputs"]).to(device=device,dtype=dtype)
        else:
            train_x_obj = [torch.tensor([])]
        if my_node["GPs"][0]["train_targets"] is not None:
            train_y_obj = torch.from_numpy(my_node["GPs"][0]["train_targets"]).to(device=device,dtype=dtype)
        else:
            train_y_obj = torch.tensor([])

        # pdb.set_trace()
        train_x_cons = torch.from_numpy(my_node["GPs"][1]["train_inputs"]).to(device=device,dtype=dtype)
        if "train_yl_cons" in my_node["GPs"][1].keys():
            train_yl_cons = my_node["GPs"][1]["train_yl_cons"].to(device=device,dtype=dtype)
        else:
            train_yl_cons = torch.from_numpy(my_node["GPs"][1]["train_targets"]).to(device=device,dtype=dtype)

        logger.info("==============================================")
        logger.info("train_x_obj:" + str(train_x_obj))
        logger.info("train_x_obj.shape:" + str(train_x_obj.shape))
        logger.info("train_y_obj:" + str(train_y_obj))
        logger.info("train_y_obj.shape:" + str(train_y_obj.shape))
        logger.info("==============================================")
        logger.info("")

        logger.info("==============================================")
        logger.info("train_x_cons:" + str(train_x_cons))
        logger.info("train_x_cons.shape:" + str(train_x_cons.shape))
        logger.info("train_yl_cons:" + str(train_yl_cons))
        logger.info("train_yl_cons.shape:" + str(train_yl_cons.shape))
        logger.info("==============================================")
        logger.info("")

        # Get logvars so far:
        logvars["regret_simple_list"] = np.ndarray.tolist(my_node["regret_simple_array"])
        logvars["regret_simple_list"] = [np.array([el]) for el in logvars["regret_simple_list"]]
        logvars["threshold_list"] = np.ndarray.tolist(my_node["threshold_array"])
        logvars["threshold_list"] = [np.array([el]) for el in logvars["threshold_list"]]
        logvars["x_next_list"] = np.ndarray.tolist(my_node["x_next_array"])
        logvars["x_next_list"] = [np.array(el) for el in logvars["x_next_list"]]

        # Report of data so far:
        logger.info("Quick report on data collected so far")
        logger.info("=====================================")
        logger.info("regret_simple_list:" + str(logvars["regret_simple_list"]))
        logger.info("threshold_list:" + str(logvars["threshold_list"]))

        # Initial iteration:
        trial_init = train_yl_cons.shape[0]

    else:

        train_x_obj, train_y_obj, train_x_cons, train_yl_cons = get_initial_evaluations(which_objective=cfg.which_objective,
                                                                                    function_obj=function_obj,
                                                                                    function_cons=function_cons,
                                                                                    cfg_Ninit_points=cfg.Ninit_points,
                                                                                    with_noise=cfg.with_noise)

        # Initial iteration:
        trial_init = 0


    # Save data in a new location:
    my_path = "./{0:s}/{1:s}_results".format(cfg.which_objective,cfg.acqui)
    path2data = generate_folder_at_path(my_path,create_folder=True)

    # ------------------
    # Initialize models:
    # ------------------
    # pdb.set_trace()
    gp_obj = GPmodel(dim=dim, train_X=train_x_obj, train_Y=train_y_obj.view(-1), options=cfg.gpmodel)
    if cfg.acqui == "EIC":
        gp_cons = GPCRmodel(dim=dim, train_x=train_x_cons.clone(), train_yl=train_yl_cons.clone(), options=cfg.gpcr_model)

    elif cfg.acqui == "EIC_standard":

        # pdb.set_trace()

        gp_cons = GPmodel(dim=dim, train_X=train_x_cons.clone(), train_Y=train_yl_cons[:,0].clone(), options=cfg.gpmodel_cons, which_type="cons")

        # Compatibility with GPCR model - We add some attributes present in GPCRmodel but missing in GPmodel
        gp_cons.threshold = torch.tensor([cfg.gpmodel_cons.threshold])
        gp_cons.train_ys = train_yl_cons[train_yl_cons[:,0] <= gp_cons.threshold,0]
        gp_cons.train_x = train_x_cons.clone()
        gp_cons.train_yl = train_yl_cons.clone()
        gp_cons._identify_stable_close_to_unstable = lambda X_sta,X_uns,top_dist,verbosity: ([],[])
        gp_cons.train_x_sorted = torch.tensor([])

    elif cfg.acqui == "EIClassi":
        ind_safe = train_yl_cons[:,1] == +1
        train_yl_cons[ind_safe,1] = +1
        train_yl_cons[~ind_safe,1] = 0
        gp_cons = GPClassifier(dim=dim, train_X=train_x_cons.clone(), train_Y=train_yl_cons[:,1].clone(), options=cfg.gpclassimodel)


    # -------------------------------------
    # Initialize acquisition function class
    # -------------------------------------
    if cfg.acqui == "EIC" or cfg.acqui == "EIC_standard":
        constraints = {1: (None, gp_cons.threshold )}
        model_list = ModelListGP(gp_obj,gp_cons)
        eic = ExpectedImprovementWithConstraints(model_list=model_list, constraints=constraints, options=cfg.acquisition_function)
    elif cfg.acqui == "EIClassi":
        model_list = [gp_obj,gp_cons]
        eic = ExpectedImprovementWithConstraintsClassi(dim=dim, model_list=model_list, options=cfg.acquisition_function)

    # pdb.set_trace()
    if (cfg.acqui == "EIC" or cfg.acqui == "EIC_standard") and model_list.train_targets[0] is not None:
        logvars["GPs"] = dict(  train_inputs=[train_inp[0] for train_inp in model_list.train_inputs],
                                train_targets=[train_tar for train_tar in model_list.train_targets])
    elif cfg.acqui == "EIClassi" and model_list[0].train_targets is not None:
        logvars["GPs"] = dict(  train_inputs=[mod.train_inputs[0] if mod.train_inputs[0] is not None else None for mod in model_list],
                                train_targets=[mod.train_targets for mod in model_list])

    # pdb.set_trace()

    # Plotting:
    if cfg.plot.plotting:
        axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj=None,axes_GPcons=None,
                                                                                    axes_GPcons_prob=None,axes_acqui=None,cfg_plot=cfg.plot)

    try:
        for trial in range(trial_init,cfg.NBOiters):
            
            msg_bo_iters = " <<< BOC Iteration {0:d} / {1:d} >>>".format(trial+1,cfg.NBOiters)
            print("\n\n")
            logger.info("="*len(msg_bo_iters))
            logger.info("{0:s}".format(msg_bo_iters))
            logger.info("="*len(msg_bo_iters))

            # Get next point:
            x_next, alpha_next = eic.get_next_point()

            # Compute simple regret:
            regret_simple = eic.get_simple_regret_cons(fmin_true=f_min)
            logger.info("Regret: {0:2.5f}".format(regret_simple.item()))
            
            if x_next is None and alpha_next is None:
                break

            if cfg.plot.plotting:
                axes_GPobj, axes_GPcons, axes_GPcons_prob, axes_acqui = plotting_tool_cons(gp_obj,gp_cons,eic,axes_GPobj,axes_GPcons,
                                                                                            axes_GPcons_prob,axes_acqui,cfg.plot,
                                                                                            xnext=x_next,alpha_next=alpha_next)

            # Logging:
            append_logging_variables(logvars,eic.eta_c,eic.x_eta_c,x_next,alpha_next,regret_simple,gp_cons.threshold)
            # pdb.set_trace()

            # Collect evaluation at xnext:
            y_new_obj   = function_obj(x_next,with_noise=cfg.with_noise)
            # yl_new_cons  = function_cons(x_next,with_noise=cfg.with_noise)
            yl_new_cons  = function_cons(x_next,with_noise=False)

            x_new_cons = x_next
            x_new_obj = x_new_cons[yl_new_cons[:,1] == +1.0,:]
            y_new_obj = y_new_obj[yl_new_cons[:,1] == +1.0]

            # Update GP model:
            if len(y_new_obj) == 0: # If there's no new data
                if gp_obj.train_inputs is None and gp_obj.train_targets is None: # and the GPobj was empty, fill with empty tensors
                    train_x_obj_new   = [torch.tensor([])]
                    train_y_obj_new   = torch.tensor([])
                else: # and the GPobj wasn't empty, don't update it
                    train_x_obj_new = gp_obj.train_inputs[0]
                    train_y_obj_new = gp_obj.train_targets
            else: # if there's new data
                if gp_obj.train_inputs is None and gp_obj.train_targets is None: # and the GPobj was empty, fill it
                    train_x_obj_new = x_new_obj
                    train_y_obj_new = y_new_obj
                else: # and the GPobj wasn't empty, concatenate
                    train_x_obj_new   = torch.cat([gp_obj.train_inputs[0], x_new_obj])
                    train_y_obj_new   = torch.cat([gp_obj.train_targets, y_new_obj])
            
            # pdb.set_trace()
            train_x_cons_new  = torch.cat([gp_cons.train_x, x_new_cons])
            train_yl_cons_new = torch.cat([gp_cons.train_yl, yl_new_cons.view(1,2)], dim=0)

            # Load GP model for f(x) and fit hyperparameters:
            gp_obj = GPmodel(dim=dim, train_X=train_x_obj_new, train_Y=train_y_obj_new.view(-1), options=cfg.gpmodel)

            # Load GPCR model for g(x) and fit hyperparameters:
            gp_cons_train_x_backup = gp_cons.train_x.clone()
            gp_cons_train_yl_backup = gp_cons.train_yl.clone()

            if cfg.acqui == "EIClassi":

                ind_safe = train_yl_cons_new[:,1] == +1
                train_yl_cons_new[ind_safe,1] = +1
                train_yl_cons_new[~ind_safe,1] = 0

                gp_cons = GPClassifier(dim=dim, train_X=train_x_cons_new.clone(), train_Y=train_yl_cons_new[:,1].clone(), options=cfg.gpclassimodel)

            elif cfg.acqui == "EIC_standard":

                gp_cons = GPmodel(dim=dim, train_X=train_x_cons_new.clone(), train_Y=train_yl_cons_new[:,0].clone(), options=cfg.gpmodel_cons, which_type="cons")

                # Compatibility with GPCR model - We add some attributes present in GPCRmodel but missing in GPmodel
                gp_cons.threshold = torch.tensor([cfg.gpmodel_cons.threshold])
                gp_cons.train_ys = train_yl_cons[train_yl_cons[:,0] <= gp_cons.threshold,0]
                gp_cons._identify_stable_close_to_unstable = lambda X_sta,X_uns,top_dist,verbosity: ([],[])
                gp_cons.train_x_sorted = torch.tensor([])
                gp_cons.train_x = train_x_cons_new.clone()
                gp_cons.train_yl = train_yl_cons_new.clone()

            elif cfg.acqui == "EIC":
                try:
                    gp_cons = GPCRmodel(dim=dim, train_x=train_x_cons_new.clone(), train_yl=train_yl_cons_new.clone(), options=cfg.gpcr_model)
                except Exception as inst:
                    logger.info("  Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
                    logger.info("  GPCR model has failed to be constructed (!!)")
                    logger.info("  This typically happens when the model the model is stuffed with datapoints, some of them rather close together,")
                    logger.info("  which causes numerical unstability that couldn't be fixed internally ...")
                    logger.info("  Trying to simply not update the GPCR model. Keeping the same number of evaluations: {0:d} ...".format(gp_cons_train_x_backup.shape[0]))
                    # gp_cons = GPCRmodel(dim=dim, train_x=gp_cons_train_x_backup, train_yl=gp_cons_train_yl_backup, options=cfg.gpcr_model) # Not needed! We keep the old one
                
            # Update the model in other classes:
            if cfg.acqui == "EIC" or cfg.acqui == "EIC_standard":
                constraints = {1: (None, gp_cons.threshold)}
                model_list = ModelListGP(gp_obj,gp_cons)
                eic = ExpectedImprovementWithConstraints(model_list=model_list, constraints=constraints, options=cfg.acquisition_function)
            elif cfg.acqui == "EIClassi":
                model_list = [gp_obj,gp_cons]
                eic = ExpectedImprovementWithConstraintsClassi(dim=dim, model_list=model_list, options=cfg.acquisition_function)


            logvars["GPs"] = [gp_obj.logging(), gp_cons.logging()]

            if cfg.acqui == "EIC_standard":
                logvars["GPs"][1].update({"train_yl_cons": gp_cons.train_yl})

            # Automatically save:
            node2write = convert_lists2arrays(logvars)
            node2write["n_rep"] = rep_nr
            node2write["ycm"] = f_min
            node2write["xcm"] = x_min
            # node2write["cfg"] = cfg # Do NOT save this, or yaml will terribly fail as it will have a cyclic graph!

            file2save = "{0:s}/data_0.yaml".format(path2data)
            logger.info("Saving while optimizing. Iteration: {0:d} / {1:d}".format(trial+1,cfg.NBOiters))
            logger.info("Saving in {0:s} ...".format(file2save))
            with open(file2save, "w") as stream_write:
                yaml.dump(node2write, stream_write)
            logger.info("Done!")

    except Exception as inst:
        logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
        traceback.print_exc()
        msg_bo_final = " <<< {0:s} failed (!) at iteration {1:d} / {2:d} >>>".format(cfg.acqui,trial+1,cfg.NBOiters)
    else:
        msg_bo_final = " <<< {0:s} finished successfully!! >>>".format(cfg.acqui)

    logger.info("="*len(msg_bo_final))
    logger.info("{0:s}".format(msg_bo_final))
    logger.info("="*len(msg_bo_final))

    node2write = convert_lists2arrays(logvars)
    node2write["n_rep"] = rep_nr
    node2write["ycm"] = f_min
    node2write["xcm"] = x_min
    # node2write["cfg"] = cfg # Do NOT save this, or yaml will terribly fail as it will have a cyclic graph!

    save_data(node2write=node2write,which_obj=cfg.which_objective,which_acqui=cfg.acqui,rep_nr=rep_nr)
