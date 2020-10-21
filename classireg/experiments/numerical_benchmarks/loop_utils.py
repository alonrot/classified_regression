import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pdb
import numpy as np
import torch
# from classireg.objectives import Hartmann6D, Michalewicz10D, Simple1D, ConsBallRegions, Branin2D, ConsCircle, WalkerObj, WalkerCons, Camel2D, Eggs2D, Shubert4D, QuadrupedObj, QuadrupedCons
from classireg.objectives import Hartmann6D, Michalewicz10D, Simple1D, ConsBallRegions, Branin2D, ConsCircle, Camel2D, Eggs2D, Shubert4D, QuadrupedObj, QuadrupedCons
from botorch.utils.sampling import draw_sobol_samples
from classireg.utils.parsing import get_logger
from classireg.utils.parse_data_collection import obj_fun_list
from omegaconf import DictConfig
logger = get_logger(__name__)
np.set_printoptions(linewidth=10000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def initialize_logging_variables():
    logvars = dict( mean_bg_list=[],
                    x_bg_list=[],
                    x_next_list=[],
                    alpha_next_list=[],
                    regret_simple_list=[],
                    threshold_list=[],
                    label_cons_list=[],
                    GPs=[])
    return logvars

def append_logging_variables(logvars,eta_c,x_eta_c,x_next,alpha_next,regret_simple,threshold=None,label_cons=None):
    if eta_c is not None and x_eta_c is not None:
        logvars["mean_bg_list"].append(eta_c.view(1).detach().cpu().numpy())
        logvars["x_bg_list"].append(x_eta_c.view(x_eta_c.shape[1]).detach().cpu().numpy())
    # else:
    #     logvars["mean_bg_list"].append(None)
    #     logvars["x_bg_list"].append(None)
    logvars["x_next_list"].append(x_next.view(x_next.shape[1]).detach().cpu().numpy())
    logvars["alpha_next_list"].append(alpha_next.view(1).detach().cpu().numpy())
    logvars["regret_simple_list"].append(regret_simple.view(1).detach().cpu().numpy())
    logvars["threshold_list"].append(None if threshold is None else threshold.view(1).detach().cpu().numpy())
    logvars["label_cons_list"].append(None if label_cons is None else label_cons.detach().cpu().numpy())
    return logvars

def get_initial_evaluations(which_objective,function_obj,function_cons,cfg_Ninit_points,with_noise):

    assert which_objective in obj_fun_list, "Objective function <which_objective> must be {0:s}".format(str(obj_fun_list))

    # Get initial evaluation:
    if which_objective == "hart6D" or which_objective == "debug6D":
        # train_x = torch.Tensor([[0.32124528, 0.00573107, 0.07254258, 0.90988337, 0.00164314, 0.41116992]]) # Randomly computed  |  initial location 1
        # train_x = torch.Tensor([[0.1859, 0.3065, 0.0886, 0.8393, 0.1175, 0.3123]]) # Randomly computed (safe as well, according to the new constraint, and robust to noise_std = 0.01)  |  initial location 2
        train_x = torch.Tensor([[0.4493, 0.6189, 0.2756, 0.7961, 0.2482, 0.9121]]) # Randomly computed (safe as well, according to the new constraint, and robust to noise_std = 0.01)  |  initial location 3
 
    if which_objective == "micha10D":
        # train_x = torch.Tensor([[0.65456088, 0.22632844, 0.50252072, 0.80747863, 0.11509346, 0.73440179, 0.06093292, 0.464906, 0.01544494, 0.90179168]]) # Randomly computed
        train_x = torch.Tensor([[0.7139, 0.6342, 0.2331, 0.8299, 0.7615, 0.8232, 0.9008, 0.1899, 0.6961, 0.3240]])

    # Get initial evaluation in g(x):
    if which_objective == "simple1D":

        # Safe/unsafe bounds according to classireg.objectives.simple1D.Simple1D.true_minimum()
        safe_area1 = torch.Tensor(([0.0],[0.0834]))
        safe_area2 = torch.Tensor(([0.4167],[1.0]))
        unsafe_area = torch.Tensor(([0.0834],[0.4167]))

        # Sample from within the bounds:
        train_x_unsafe = draw_sobol_samples(bounds=unsafe_area,n=cfg_Ninit_points.unsafe,q=1).squeeze(1) # Get only unstable evaluations
        train_x_area2 = draw_sobol_samples(bounds=safe_area2,n=cfg_Ninit_points.safe,q=1).squeeze(1) # Get only stable evaluations

        # Concatenate:
        train_x = torch.cat([train_x_unsafe,train_x_area2])

    if which_objective == "branin2D":
        # train_x = draw_sobol_samples(bounds=torch.Tensor(([0.0]*2,[1.0]*2)),n=cfg_Ninit_points.total,q=1).squeeze(1)
        train_x = torch.tensor([[0.6255, 0.5784]])

    if which_objective == "camel2D":
        train_x = torch.tensor([[0.9846, 0.0587]])        

    if which_objective == "quadruped8D":
        # train_x = torch.tensor([[0.9846, 0.0587, 0.9846, 0.9846, 0.9846, 0.9846, 0.9846, 0.9846]]) 
        # train_x = torch.tensor([[1,0,0,0,1,1,0.666,0.666]]) # Correpsonds to 21.07.2020 Recording: after 13:52 (5 experiments)
        # train_x = torch.tensor([[0.0,0.0,1.0,1.0]]) # Correpsonds to 21.07.2020, 23.07.2020, 27.07.2020 Same as above but without kp_joint_min and kd_joint_max
        train_x = torch.tensor([[0.2,0.2,0.6666666666666666,0.6666666666666666,1.0]]) # Correpsonds to 29.07.2020

    if which_objective == "eggs2D":
        train_x = torch.Tensor([[0.5578, 0.0558]])

    if which_objective == "walker":
        # train_x = torch.tensor([[0.5  ,0.25, 0.5,  0.25, 0.5,  0.75]]) + 0.12*torch.randn(6) # Found with brute force, x_in \in 10**[-0.5,0.5], multiplicative; Stable 94 / 100
        # train_x = torch.tensor([[0.67110407, 0.24342352, 0.71659071, 0.37363523, 0.52991535, 0.49756885]]) # Stable 65 / 100 | 0.12
        train_x = torch.tensor([[0.26790537, 0.3768979 , 0.49344913, 0.18835246, 0.57790874, 0.7599986 ]]) # Stable 40 / 100 | 0.12 (never actually used)
        # train_x = torch.tensor([[0.38429936, 0.26406187, 0.59825079, 0.24133024, 0.43413793, 0.77263459]]) # Stable 97 / 100 | 0.12 (never actually used)

    if which_objective == "shubert4D":
        train_x = torch.tensor([[0.7162, 0.3331, 0.8390, 0.8885]]) # 11.1146

    # Evaluate objective and constraint(s):
    # NOTE: Do NOT change the order!!

    # Get initial evaluations in f(x):
    train_y_obj = function_obj(train_x,with_noise=with_noise)

    # Get initial evaluations in g(x):
    train_x_cons = train_x
    train_yl_cons = function_cons(train_x_cons,with_noise=False)

    # Check that the initial point is stable in micha10D:
    # pdb.set_trace()

    # Get rid of those train_y_obj for which the constraint is violated:
    train_y_obj = train_y_obj[train_yl_cons[:,1] == +1]
    train_x_obj = train_x[train_yl_cons[:,1] == +1,:]

    logger.info("train_x_obj: {0:s}".format(str(train_x_obj)))
    logger.info("train_y_obj: {0:s}".format(str(train_y_obj)))
    logger.info("train_x_cons: {0:s}".format(str(train_x_cons)))
    logger.info("train_yl_cons: {0:s}".format(str(train_yl_cons)))

    return train_x_obj, train_y_obj, train_x_cons, train_yl_cons

def get_objective_functions(which_objective):

    assert which_objective in obj_fun_list, "Objective function <which_objective> must be {0:s}".format(str(obj_fun_list))

    if which_objective == "hart6D" or which_objective == "debug6D":
        func_obj = Hartmann6D(noise_std=0.01)
        dim = 6
        function_cons = ConsBallRegions(dim=dim,noise_std=0.01)
    if which_objective == "micha10D":
        dim = 10
        func_obj = Michalewicz10D(noise_std=0.01)
        function_cons = ConsBallRegions(dim=dim,fac_=10.0)
    if which_objective == "simple1D":
        func_obj = Simple1D()
        dim = 1
        function_cons = ConsBallRegions(dim=dim)
    if which_objective == "branin2D":
        func_obj = Branin2D(noise_std=0.01)
        function_cons = ConsCircle(noise_std=0.01)
        dim = 2
    if which_objective == "quadruped8D":
        # dim = 8
        # dim = 4
        dim = 5
        func_obj = QuadrupedObj(dim=dim)
        function_cons = QuadrupedCons(func_obj)
    if which_objective == "camel2D":
        func_obj = Camel2D(noise_std=0.01)
        dim = 2
        function_cons = ConsBallRegions(dim=dim,noise_std=0.01)
    if which_objective == "eggs2D":
        dim = 2
        func_obj = Eggs2D(noise_std=0.01)
        function_cons = ConsBallRegions(dim=dim,noise_std=0.01)
    if which_objective == "walker":
        path2model = "./../../objectives/walker_env/2020-04-18_21-28-42-walker2d_s0"
        max_steps_ep_replay = 1500 # steps
        num_episodes = 20 # compute the mean over these number of episodes. Need more than 7 stable episodes to consider the average as stable
        manual_constraint_thres = None
        # manual_constraint_thres = 0.25
        dim = 6
        func_obj = WalkerObj(dim=dim,path2model=path2model,max_steps_ep_replay=max_steps_ep_replay,
                            num_episodes=num_episodes,which_env2load="real",
                            manual_constraint_thres=manual_constraint_thres,render=False)
        function_cons = WalkerCons(func_obj)
    if which_objective == "shubert4D":
        dim = 4
        func_obj = Shubert4D(noise_std=0.01)
        function_cons = ConsBallRegions(dim=dim,noise_std=0.01)


    # Get the true minimum for computing the regret:
    x_min, f_min = func_obj.true_minimum()
    logger.info("<<< True minimum >>>")
    logger.info("====================")
    logger.info("  x_min:" + str(x_min))
    logger.info("  f_min:" + str(f_min))

    return func_obj, function_cons, dim, x_min, f_min


