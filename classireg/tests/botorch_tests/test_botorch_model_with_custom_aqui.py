from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from botorch.fit import fit_gpytorch_model
import random
import numpy as np
from ax import ParameterType, RangeParameter, SearchSpace
from ax import SimpleExperiment
from ax.modelbridge import get_sobol
from ax.modelbridge.factory import get_botorch

# For custom acqui:
import math

import torch
from torch import Tensor
# from typing import Optional

# from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, ScalarizedObjective, AcquisitionObjective, IdentityMCObjective
from botorch.models.model import Model
# from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
from botorch.sampling.samplers import MCSampler

# from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
# from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Optional, Tuple



# from botorch.acquisition import AnalyticAcquisitionFunction

class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API
    
    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class qScalarizedUpperConfidenceBound(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        objective = IdentityMCObjective()
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)  # n x b x q x o
        scalarized_samples = samples.matmul(self.weights)  # n x b x q
        mean = posterior.mean  # b x q x o
        scalarized_mean = mean.matmul(self.weights)  # b x q
        ucb_samples = (
            scalarized_mean
            + math.sqrt(self.beta * math.pi / 2)
            * (scalarized_samples - scalarized_mean).abs()
        )
        return ucb_samples.max(dim=-1)[0].mean(dim=0)

class ScalarizedUpperConfidenceBound(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor,
        X_pending: Optional[Tensor] = None,
        # objective_weights: Tensor,
        # outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        X_observed: Optional[Tensor] = None,
        # X_pending: Optional[Tensor] = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model)
        self.maximize = maximize
        self.X_observed = X_observed
        # self.X_pending = X_pending
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))
        # self.register_buffer("X_pending", torch.as_tensor(X_pending))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the Upper Confidence Bound on the candidate set X using scalarization

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        self.beta = self.beta.to(X)
        batch_shape = X.shape[:-2]
        posterior = self.model.posterior(X)
        means = posterior.mean.squeeze(dim=-2)  # b x o
        scalarized_mean = means.matmul(self.weights)  # b
        covs = posterior.mvn.covariance_matrix  # b x o x o
        weights = self.weights.view(1, -1, 1)  # 1 x o x 1 (assume single batch dimension)
        weights = weights.expand(batch_shape + weights.shape[1:])  # b x o x 1
        weights_transpose = weights.permute(0, 2, 1)  # b x 1 x o
        scalarized_variance = torch.bmm(
            weights_transpose, torch.bmm(covs, weights)
        ).view(batch_shape)  # b
        delta = (self.beta.expand_as(scalarized_mean) * scalarized_variance).sqrt()
        if self.maximize:
            return scalarized_mean + delta
        else:
            return scalarized_mean - delta

def get_qscalarized_UCB(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: None,
    ) -> AcquisitionFunction:
    # return ScalarizedUpperConfidenceBound(model=model, beta=0.2, weights=objective_weights)
    return qScalarizedUpperConfidenceBound(model=model, beta=0.2, weights=objective_weights)

def branin(parameterization, *args):
    x1, x2 = parameterization["x1"], parameterization["x2"]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    # let's add some synthetic observation noise
    y += random.normalvariate(0, 0.1)
    return {"branin": (y, 0.0)}

def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):

    # user: Xs is a list. Thereby, the zero

    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # import pdb; pdb.set_trace()

    # Some hyperrparameters
    print(model.covar_module.base_kernel.lengthscale)
    print(model.mean_module.constant)
    print(model.covar_module.outputscale)

    # TODO:
    # 1) Set hyperparameter priors
    # 2) Set hyperparameter boundaries
    # 3) Activate/deactivate which hyperparameters are learned and which aren't
    # 4) Activate/deactivate NUTS/ML-II
    # 5) Use a custom model that returns the gradmean and gradcov, and selects p(Y,f*,\Delta f*)

    return model

# def initialize_model():

#     # generate synthetic data
#     X = torch.rand(20, 2)
#     Y = torch.stack([torch.sin(X[:, 0]), torch.cos(X[:, 1])], -1)

#     # construct and fit the multi-output model
#     gp = SingleTaskGP(X, Y)
#     mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#     fit_gpytorch_model(mll);

#     return gp


search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)

exp = SimpleExperiment(
    name="test_branin",
    search_space=search_space,
    evaluation_function=branin,
    objective_name="branin", # This name has to coincide with the name of the function to call
    minimize=True,
)


sobol = get_sobol(exp.search_space)
exp.new_batch_trial(generator_run=sobol.gen(5))


for i in range(5):
    print(f"Running optimization batch {i+1}/5...")

    # user added:
    # get_botorch is a self-contained macro-framework that does everything for you:
    # 1) It fits the GP model to the new data
    # 2) If optimizes the acquisition function and provides the next candidate
    # 3) It evaluates such candidate (next point)
    # 4) It returns the best suggestion (best guess)
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=_get_and_fit_simple_custom_gp,
        acqf_constructor=get_qscalarized_UCB,
    )
    batch = exp.new_trial(generator_run=model.gen(n=1)) # See https://ax.dev/versions/0.1.3/api/_modules/ax/models/torch/botorch.html#BotorchModel.gen
    # Not sure if it's the above URL or this one: https://ax.dev/versions/0.1.3/api/models.html?highlight=gen#ax.models.torch_base.TorchModel.gen
    # An initialized acquisition function can be passed in as model_gen_options[“acquisition_function”].
    # gen() -> Generate new cndidates.
    # n: Number of candidates to generate.

    # user TODO: See https://ax.dev/api/_modules/ax/modelbridge/factory.html#get_botorch
    # 1) Pass a custom acquisition function constructor
    # 2) Maybe change the optimizer
    # 3) How do we control the number of batches being used, when using the MC versions (the only ones that actually work...)
    # 4) Update the data without the need of re-instantating the model:
    	# https://ax.dev/versions/0.1.3/api/models.html?highlight=gen#ax.models.torch.botorch.BotorchModel.gen

    print(batch)
    
print("Done!")
