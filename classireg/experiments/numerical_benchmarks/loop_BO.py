import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pdb
import numpy as np
import torch
from classireg.experiments.numerical_benchmarks.loop_utils import initialize_logging_variables,append_logging_variables,get_initial_evaluations,get_objective_functions
from classireg.utils.plotting_collection import plotting_tool_uncons
from classireg.acquisitions.expected_improvement import ExpectedImprovementVanilla
from classireg.utils.parsing import convert_lists2arrays, save_data, get_logger
from omegaconf import DictConfig
from classireg.models.gpmodel import GPmodel
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
    
    train_x_obj, train_y_obj, train_x_cons, train_yl_cons = get_initial_evaluations(which_objective=cfg.which_objective,
                                                                                    function_obj=function_obj,
                                                                                    function_cons=function_cons,
                                                                                    cfg_Ninit_points=cfg.Ninit_points,
                                                                                    with_noise=cfg.with_noise)

    if torch.any(train_yl_cons[:,1] == -1):
        assert train_yl_cons.shape[0] == 1, "Case with more than one unsafe evaluations not implemented yet (!)"
        if cfg.acqui == "EI_heur_high": # Replace the unstable observations with a pre-defined heuristic value
            train_y_obj = torch.tensor([cfg.cost_heur_high])
            train_x_obj = train_x_cons
            logger.info("Fixed heuristic cost (high): {0:f}".format(cfg.cost_heur_high))
        elif cfg.acqui == "EI_heur_low": # Replace the unstable observations with a pre-defined heuristic value
            train_y_obj = torch.tensor([cfg.cost_heur_low])
            train_x_obj = train_x_cons
            logger.info("Fixed heuristic cost (low): {0:f}".format(cfg.cost_heur_low))
        else:
            raise NotImplementedError("EI method cannot be applied in this case because no initial safe value has been obtained yet")

    gp_obj = GPmodel(dim=dim,train_X=train_x_obj, train_Y=train_y_obj.view(-1), options=cfg.gpmodel)

    ei = ExpectedImprovementVanilla(model=gp_obj, options=cfg.acquisition_function)

    logvars = initialize_logging_variables()
    
    if gp_obj.train_targets is not None:
        logvars["GPs"] = dict(  train_inputs=[train_inp[0] for train_inp in gp_obj.train_inputs],
                                train_targets=[train_tar for train_tar in gp_obj.train_targets])

    # Plotting:
    if cfg.plot.plotting:
        axes_GPobj, axes_acqui = plotting_tool_uncons(gp_obj,ei,axes_GPobj=None,axes_acqui=None,plot_eta=False)

    # Label:
    label_cons = train_yl_cons[:,1]

    try:

        # average over multiple trials
        for trial in range(cfg.NBOiters):
            
            msg_bo_iters = " <<< BO Iteration {0:d} / {1:d} >>>".format(trial+1,cfg.NBOiters)
            print("\n\n")
            logger.info("="*len(msg_bo_iters))
            logger.info("{0:s}".format(msg_bo_iters))
            logger.info("="*len(msg_bo_iters))

            # Get next point:
            x_next, alpha_next = ei.get_next_point()

            # Compute simple regret:
            regret_simple = ei.get_simple_regret(fmin_true=f_min)
            if len(regret_simple) > 0:
                logger.info("Regret: {0:2.5f}".format(regret_simple.item()))
            else:
                logger.info("Regret can't be computed because no safe point has been found yet")
            
            if x_next is None and alpha_next is None:
                break

            if cfg.plot.plotting:
                axes_GPobj, axes_acqui = plotting_tool_uncons(gp_obj,ei,axes_GPobj,axes_acqui,xnext=x_next,alpha_next=alpha_next,Ndiv=201,plot_eta=False)

            # Collect evaluation at xnext:
            y_new_obj   = function_obj(x_next,with_noise=cfg.with_noise)
            yl_new_cons  = function_cons(x_next,with_noise=cfg.with_noise)
            label_cons_new = yl_new_cons[:,1]
            label_cons = torch.cat([label_cons, label_cons_new])
            x_new_obj = x_next

            # Update GP model:
            if gp_obj.train_inputs is None and gp_obj.train_targets is None: # and the GPobj was empty, fill it
                train_x_obj_new = x_new_obj
                train_y_obj_new = y_new_obj
                print(  "We commented out the line below; make sure the code can continue. Later on we are using label_cons. Is it ok?\n \
                        ")
                pdb.set_trace()
                # train_y_obj_new[label_cons_new == -1] = torch.max(y_new_obj) # Replae the unstable observations with the highest value so far
            else: # and the GPobj wasn't empty, concatenate
                train_x_obj_new   = torch.cat([gp_obj.train_inputs[0], x_new_obj])
                train_y_obj_new   = torch.cat([gp_obj.train_targets, y_new_obj])
                # train_y_obj_new[label_cons == -1] = torch.max(train_y_obj_new) # Replace the unstable observations with the highest value so far

            if cfg.acqui == "EI": # Replace the unstable observations with the highest value so far
                train_y_obj_new[label_cons == -1] = torch.max(train_y_obj_new)
                logger.info("Adaptive heuristic cost: {0:f}".format(torch.max(train_y_obj_new).item()))
            elif cfg.acqui == "EI_heur_high": # Replace the unstable observations with a pre-defined heuristic value
                train_y_obj_new[label_cons == -1] = cfg.cost_heur_high
                logger.info("Fixed heuristic cost (high): {0:f}".format(cfg.cost_heur_high))
            elif cfg.acqui == "EI_heur_low": # Replace the unstable observations with a pre-defined heuristic value
                train_y_obj_new[label_cons == -1] = cfg.cost_heur_low
                logger.info("Fixed heuristic cost (low): {0:f}".format(cfg.cost_heur_low))

            logger.info("Nr. unsafe evaluations: {0:d}".format(torch.sum(label_cons == -1).item()))

            # Logging:
            append_logging_variables(logvars,ei.eta,ei.x_eta,x_next,alpha_next,regret_simple,label_cons=label_cons)

            # Load GP model and fit hyperparameters:
            gp_obj = GPmodel(dim=dim,train_X=train_x_obj_new, train_Y=train_y_obj_new.view(-1), options=cfg.gpmodel)

            # Update the model in other classes:
            ei = ExpectedImprovementVanilla(model=gp_obj, options=cfg.acquisition_function)

            logvars["GPs"] = [gp_obj.logging(), None]

    except Exception as inst:
        logger.info("Exception (!) type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
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

