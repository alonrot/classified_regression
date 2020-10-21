

# TODO

1. When the input location points are too close one another, the hyperparameter search breaks. Possible solutions:
	0. **Adopted temporal solution: restricted the hyperpriors range to [0.,75.] of the inverse CDF**
	1. DONE: Try to make EP more robust -> Fixed the prior covariance matrix that enters EP solved many of the caused issues...
	2. Use the *quick fix* about changing the noise/threshold
	3. 	Same as above, but use heteroscedastic noise, and let it be learned together with the other hyperpars.
			Maybe, research a bit about GPs and heteroscedastic noise. Maybe there is some popular solution we can use.
			Problem: The more evaluations we have, the more noise parameters we'll have, so it scales badly...
			Maybe we can just **detect the points that are too close and only act on those** using our tool.
2. Put possible everything related to the hyperpriors inside the model. Right now the *hyperpriors_support*
		is defined inside mll while the hyperpriors are defined in the top level script.
3. 

### Corner cases EIC
1. ~~When there is no feasible evaluation yet, switch to the pure constraint maximization~~
2. ~~Implement looking for the constrained minimum of the posterior mean~~
4. EP shows issues when two points are too close one another
	4.1  HELPED Eliminate points that are *extremely* close to each other, to avoid numerical unstability. Explain this as "inducing points"
			A. Points that are too close dist = 0.02
			B. First time the EP posterior is computed: when optimizing hyperpars -> dist = ??
			C. Second time the EP posteroir is computed: when doing predictions -> dist = 0.5*min(lengthscales)
	4.2  Ways of fixing the covariance matrices:
				* Or maybe we could fix the covariance matrix inside gpytorch
				* Pass the self.gauss_tools.fix_singular_matrix() to line 280 in the gpcr model
				* Pass it to the self.covariance_posterior in line 350
	4.5  HELPED Increasing the likelihood noise
5. EIC shows pathologies when we start with an unstable constraint evaluation and the rest are close to zero because the prior mean is at zero.
	5.1 HELPED kind of solved it by playing around with the hyperprior in the threshold
6. ~~Test what happens when two consecutive unstable measurements are obtained~~
8. ~~The regret shows as negative~~
7. Test and fix for multiple dimensions optimization
9. If the algorithm crashes, have a way to restart it: Will need it during robot experiments 
	1. Make sure that the data is properly logged
	2. parse_data_collection.py -> It must do the right thing
11. ~~See how to initialize the threshold when no stable evaluations are present~~
12. ~~Use yaml files~~
13. The noise matrix is modified heuristically. Think about including it in the hyperparameter optimization problem... 
		The dimensionality of the parameter space woudl increase accordingly
14. If we end up modifying the noise matrix, maybe show the affected hetersocedastic components in the GPCRmodel.display_hyperparameters()
13. ~~Define the hyperparameters as properties of the model (setter and getter), to emphasize that other classes will be using them~~
16. ~~Setting a top-level torch seed doesn't seem to work...~~
17. Finding the constrained GP mean still fails ... (!!)
18. ~~Fix, in EIC, the best_f and do it as the worst constrained observation.~~
19. If we start with an unstable evaluation, nor EIC neither EI will work
20. Proof convergence of the estimated threshold to the optimal threshold when the data collected is \inf



Try more functions:

http://benchmarkfcns.xyz/benchmarkfcns/threehumpcamelfcn.html
scores = (2 * X .^ 2) - (1.05 * (X .^ 4)) + ((X .^ 6) / 6) + X .* Y + Y .^2;

http://benchmarkfcns.xyz/benchmarkfcns/goldsteinpricefcn.html

http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html

https://arxiv.org/pdf/1308.4008.pdf



ssh alonso@pcics01.mpi-stuttgart.mpg.de

Email about authentication in mpi stuttgart:
https://mail.google.com/mail/u/0/#search/alex/FMfcgxwGDDjqqRjDhcKhJPTBlxqJshQZ


ssh amarcovalle@login.cluster.is.localnet


https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d.py



# Added manual constraint threshold to walker
-> Running experiments on cluster with thres=0.5 and EI_heur_high DONE
-> Repeated the above with thres=0.25

# Experiments on cluster with eggs2D EI:
-> repeating these experiments for 100 iterations for consistency


BEFORE CLUSTER
==============
* walker.yaml -> Niters to 50
* loop_BO.py remove all pdb.settarce
* loop_utils.py -> num_episodes = 20 (increase)
* Make sure EI_heur_high is on

Can't really use EI in walker because the initial evaluations are unstable and there's no pre-existing value to give tot he unstable cases



condor_submit_bid 500 config/cluster/launch.sub


# Vamos a ver
* Plot the results from Camel and figure out the thing about the negative regrets
* Plot the results from walker and figure out why such a lagre variance
* The new shubert function is suposed to be running in cluster, although caondor_q amarcovalle doen't spit antyhtinf

* Examine: total number of failures incurred by each algorithm. Hopefully EIC will have less failures.
* For walker:
	* Motivation: super hard problem; an initial grid search indicated that only 26 / 15625 points are safe approx. 0.2% (filename = "walker_data_6D_brute_force_multipliying.yaml"), with 6^5 = 15625.
	* Initial point near stable policies (warm start), 50 iterations top, repeating 100 times.
	* Only 25% of the experiments were able to find at least one stable policy. Stable policy means that the robot is stabilizable >= 18/20 times
	* Among that 25%, XX% found the first stable poliy after 10 experiments.
		* Box plot: _When_ was the first stable policy found
	* Coompare, with box plots / barplots, the initially found controller and the final controller
		* Start with a historgram, to study the kurtosis
		* Alternatively: Use the utility proposed in PESC
	


python -m spinup.run ppo --env Walker2d-v2 --exp_name walker2d --dt --gamma 0.999 --epochs 2000 --num_cpu 4


pip install gym
download mujoco following Alex' instricutions in whatsapp:
	1) Download it from https://www.roboti.us/index.html (mujoco200 macos)
	2) Place the contents of the downloaded folder into /Users/alonrot/.mujoco/mujoco200
	3) Place the key into /Users/alonrot/.mujoco
	4) add this to ~/.bash_profile: LD_LIBRARY_PATH=/Users/alonrot/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH            ~


pip install mujoco_py: This will throw 2 errors in macOS catalina:
1) Cannot find gcc: brew install gcc
2) Window pops up during installation saying libmujoco200.dylib cannot be opened because the developer cannot be verified. We need to manually grant iTerm permission to make changes. For this, System Preferences -> Security & Privacy -> (Privace Tab) -> Developer Tools
There, add manually "iTerm" by clicking on the "+" icon. If you have iTerm open, you'll be required to close it.
After this, mujoco should be installed and running

pip install joblib

Install spinningup:
https://spinningup.openai.com/en/latest/user/installation.html




To avoid using spinning, copy MLPGaussianActor and MLPActorCritic from
/Users/alonrot/MPI/WIP_projects/gymwalker/spinningup/spinup/algos/pytorch/ppo.py 
somewhere locally and make sure we save only instances of those classes








# Walker: attempt to improve the results
<!-- * Select a threshold way higer, keeping the same hyperpars and re-run EI_higher_thres -> Re-running this right now 6 Jun 00:25 -> DONE, it's better -->
<!-- * Re-run EIC by having an optimistic prior mean (change the constraint values, etc.). Re-running this right now 6 Jun 00:20 -> DONE, it's worse -->


# Walker
<!-- * Test the learned opti threshold by running many experiments with some learned controller. -> Data is available! -->


<!-- # Walker -->
<!-- * Aren't we saving the controller parametrizations? Maybe we need to convertt hem as well. -->

<!-- * Using the same initial point: 
	* Comapring against EI_heur_high, with cost_heur_high: -1.0
	* Using pessimistic mean -->



<!-- Reimplement expected_improvement_with_constraints_gpclassi withou the model list thing -->
<!-- 14.06 18:26 Running EIClassi on cluster
						Bug in EIC data. Something wrong with the logging of train_ys_list


15.06 Running EIC on cluster with 
	train_x = torch.tensor([[0.38429936, 0.26406187, 0.59825079, 0.24133024, 0.43413793, 0.77263459]]) # Stable 97 / 100 | 0.12 (never actually used)

15.06 Running EIClassi and EI_heur_high with
	train_x = torch.tensor([[0.26790537, 0.3768979 , 0.49344913, 0.18835246, 0.57790874, 0.7599986 ]]) # Stable 40 / 100 | 0.12 (never actually used)
EI_heur_high with
	cost_heur_high: 20.0 # Upper bound # Used in

 -->

* Redoing EIC on cluster with
	train_x = torch.tensor([[0.26790537, 0.3768979 , 0.49344913, 0.18835246, 0.57790874, 0.7599986 ]]) # Stable 40 / 100 | 0.12 (never actually used)
	Using optimistic mean this time ...
	(never retrieved the data actually...)
* Redoing the above thing, but after having fixed the classireg model. Started on 30.06.2020 00:33


# Experiments on cluster with eggs2D EIC: 
	good! copied to 20200630002120

# Experiments on cluster with micha10D EIC: 
	running...

# Experiments on cluster with hart6D EIC: 
	running...


Treat it as a small idea that poses two benefits:
	1) I don't need to think about a heuristic cost if I was to solve the constrained problem with BO
	2) IF solving it BOC:
		- I could solve it with EIC + GPClassi

	3) 	We could go one step further: we learn a ball of a specific diameter that determines the region of attraction ...
			That's what we're learning with our GPCR model, actually ...
			We can pose the simple example of only one state ...
			For the walker problem we're sure we're learning the boundaries of a region of attraction.
			For the same controller parametrization, for every initial state, we leave/stay inside
			the region of attraction. We just happen to only be looking at one of the states: the tilting angle.

	4) This ideas will only be saved by one thing: A real application where learning the constraint threshold is necessary and GPC performs worse.
	5) The results need to be improved, overall. How? No idea...
	6) We need some convergence proof or something for the threshold...
		* Study the threshold posterior with a plot. Easy to do. Just run EIC2 for a while on some constrained problem and see if the ML estimate eventually converges to the true threshold or not. If yes, then maybe there's a way to prove it.

	Cite the work
	"Safe non-smooth black-box optimization with application to policy search" -> Here the constraint is a circle and they know the threshold

<!-- conda deactivate;conda deactivate;cd ~/lightspace;source workspace/devel/setup.bash -->


WALKER CONTRAINT DEFINITION
* We need to define it positive, so that the piorir zero mean is always below the estimated threshold; this way we have an optimistic exploration, as unforeseen regions will be regarded as probably safe regions


---

### Running:

* pcics02:
	* tmux 0: brute force 6D multiply weights/bias of all layers
	* tmux 1: Retraining PPO 2000 epochs
	* tmux 2: search 6D multiply weights/bias of all layers, SLSQP


---



https://youtu.be/TJzKH3PUPWg?t=597











---

[__main__] =============================
[__main__]  <<< BO Iteration 35 / 35 >>>
[__main__] =============================
[classireg.acquisitions.acquisition_base_cons] Finding min_x mu(x|D) s.t. Pr(g(x) <= 0) > 0.90
[classireg.acquisitions.acquisition_base_cons] Done!
[classireg.acquisitions.expected_improvement_with_constraints] Computing next candidate by maximizing the acquisition function ...
[classireg.acquisitions.expected_improvement_with_constraints] Generating random restarts ...
[classireg.acquisitions.expected_improvement_with_constraints] Using nlopt ...
[classireg.utils.optimize] Optimizing EIC acquisition with nlopt | Nrestarts_local = 10
[classireg.utils.optimize] Optimizer restarted 2 / 10 times
[classireg.utils.optimize] Optimizer restarted 4 / 10 times
[classireg.utils.optimize] Optimizer restarted 6 / 10 times
[classireg.utils.optimize] Optimizer restarted 8 / 10 times
[classireg.utils.optimize] Optimizer restarted 10 / 10 times
[classireg.utils.optimize] The optimization problem was succesfully solved!
[classireg.utils.optimize] Optimizing EIC acquisition with nlopt | Done!
[classireg.acquisitions.expected_improvement_with_constraints] Done!
[classireg.acquisitions.expected_improvement_with_constraints] xnext: [[1.         0.76444644]]
[classireg.acquisitions.expected_improvement_with_constraints] alpha_next: 0.71
[classireg.acquisitions.expected_improvement_with_constraints] self.x_eta_c: tensor([[0.4827, 0.3821]])
[classireg.acquisitions.expected_improvement_with_constraints] self.eta_c: tensor([11.6129], grad_fn=<ViewBackward>)
[classireg.acquisitions.expected_improvement_with_constraints] self.best_f: tensor([11.6129], grad_fn=<ViewBackward>)
[__main__] Regret: 11.24324


[classireg.models.gpmodel] ### Initializing GP model for objective f(x) ###
[classireg.models.gpmodel] Fitting GP model f(x) ...
[classireg.models.gpmodel] -------------------------
[classireg.models.gpmodel]   Re-optimized hyperparameters
[classireg.models.gpmodel]   ----------------------------
[classireg.models.gpmodel]     Outputscale (stddev) | 48.1397
[classireg.models.gpmodel]     Lengthscale(s)       | [0.17961721 0.16363178]


[classireg.models.gpcr_model] ### Initializing GPCR model for constraint g(x) ###
[classireg.models.gpcr_model] Updating after adding new data point...
[classireg.models.gpcr_model] hyperpars_bounds:[[0.001, 0.001, 0.001, 0.001], [0.18175060542624433, 0.18175060542624433, 7.840804120585122, 5.38526905777939]]
@GaussianTools.fix_singular_matrix(): singular_mat needs to be fixed
what2fix: Fixing self.prior_cov while restarting EP ...
largest_eig:  0.006998752
max_ord:  -2.0
> /Users/alonrot/MPI/WIP_projects/classified_regression/classireg/utils/gaussian_tools.py(231)fix_singular_matrix()
-> raise ValueError("Matrix could not be fixed. Something is really wrong here...")
(Pdb) singular_mat_new
array([[7.76388729e-03, 3.78957316e-06, 2.31389622e-05, ..., 2.29303623e-24, 0.00000000e+00, 5.05203965e-26],
       [3.78957316e-06, 7.76388729e-03, 2.44332652e-04, ..., 1.04791486e-19, 0.00000000e+00, 2.26106207e-30],
       [2.31389622e-05, 2.44332652e-04, 7.76388729e-03, ..., 7.66683135e-22, 0.00000000e+00, 4.31197274e-28],
       ...,
       [2.29303623e-24, 1.04791486e-19, 7.66683135e-22, ..., 7.76388729e-03, 0.00000000e+00, 0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ..., 0.00000000e+00, 7.76388729e-03, 2.15391551e-29],
       [5.05202147e-26, 2.26107918e-30, 4.31197274e-28, ..., 0.00000000e+00, 2.15388286e-29, 7.76388729e-03]], dtype=float32)
(Pdb) la.cholesky(singular_mat_new)
array([[ 8.8112921e-02,  0.0000000e+00,  0.0000000e+00, ...,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 4.3008142e-05,  8.8112913e-02,  0.0000000e+00, ...,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 2.6260578e-04,  2.7728213e-03,  8.8068895e-02, ...,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       ...,
       [ 2.6023835e-23,  1.1892864e-18, -2.8738891e-20, ...,  8.8112921e-02,  0.0000000e+00,  0.0000000e+00],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, ...,  0.0000000e+00,  8.8025756e-02,  0.0000000e+00],
       [ 5.7335762e-25, -2.5419617e-28,  3.1944890e-27, ...,  4.2038954e-45,  3.8746456e-25,  8.8112921e-02]], dtype=float32)
(Pdb) la.eigvals(singular_mat_new)
array([0.0080097 , 0.00751879, 0.00776318, 0.00810975, 0.00741915, 0.00781703, 0.00771075, 0.00783348, 0.00769429, 0.00778018, 0.00774759, 0.00777068, 0.00775709, 0.00776392, 0.00776385, 0.00776208, 0.00776457, 0.00776403, 0.00776374, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389, 0.00776389], dtype=float32)
(Pdb) la.det(singular_mat_new)
0.0
(Pdb) aaa = la.det(singular_mat_new)
(Pdb) aaa == 0.0
True
(Pdb)

* la.det() -> Even when the matrix is correctly fixed, the determinant can still be zero if the eignevalues are too small. This issue gets worse as the matrix grows, because more eigenvalues with low values will exist. We need to somehow address this issue. This error is about fixing the matrix right before entering EP (I think). So, maybe checking det() here is not the best idea, and it only makes sense to fix it later on within EP.
