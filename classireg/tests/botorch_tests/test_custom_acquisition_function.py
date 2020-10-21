import math

import torch
from torch import Tensor
from typing import Optional

from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition import AcquisitionFunction
# from botorch.acquisition.monte_carlo import MCAcquisitionFunction, IdentityMCObjective
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective, ScalarizedObjective, AcquisitionObjective, IdentityMCObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

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
        maximize: bool = True,
    ) -> None:
        objective = ScalarizedObjective(weights=torch.tensor([0.1, 0.5]),offset=0.0)
        super().__init__(model=model,objective=objective)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))

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



def initialize_model():

    # generate synthetic data
    X = torch.rand(20, 2)
    Y = torch.stack([torch.sin(X[:, 0]), torch.cos(X[:, 1])], -1)

    # construct and fit the multi-output model
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll);

    return gp


# See https://botorch.org/api/_modules/botorch/acquisition/objective.html#MCAcquisitionObjective
acqui_obj_mc = IdentityMCObjective()

# acqui_onj_an = ScalarizedObjective(weights=torch.tensor([0.1, 0.5]),offset=0.0)

def test_qUCB():

    gp = initialize_model()

    # construct the acquisition function
    qSUCB = qScalarizedUpperConfidenceBound(gp, beta=0.1, weights=torch.tensor([0.1, 0.5]))

    # evaluate on single q-batch with q=3
    qSUCB(torch.rand(3, 2))


def test_aUCB():

    gp = initialize_model()

    # construct the acquisition function
    SUCB = ScalarizedUpperConfidenceBound(gp, beta=0.1, weights=torch.tensor([0.1, 0.5]))
    
    # evaluate on single point
    SUCB(torch.rand(1, 2))
    
    # batch-evaluate on 3 points
    SUCB(torch.rand(3, 1, 2))


def get_scalarized_UCB(
    model: Model,
    objective_weights: Tensor,
    **kwargs: None,
    ) -> AcquisitionFunction:
    return ScalarizedUpperConfidenceBound(model=model, beta=0.2, weights=objective_weights)


if __name__ == "__main__":

    test_qUCB()
    # test_aUCB()
