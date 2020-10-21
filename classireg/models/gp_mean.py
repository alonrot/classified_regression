from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
import pdb

class GPmean(AnalyticAcquisitionFunction):
	def __init__(self, model: Model) -> None:
		super().__init__(model=model, objective=ScalarizedObjective(weights=Tensor([1.0])))

	@t_batch_mode_transform(expected_q=1)
	def forward(self, X: Tensor) -> Tensor:
		"""Evaluate scalarized qUCB on the candidate set `X`.

		Args:
		    X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
		        design points each.

		Returns:
				Tensor: A `(b)`-dim Tensor at the given design points `X`.
		"""

		# Get posterior GP conditioned on data, to obtain p(u|D)
		posterior_X = self.model(X.squeeze(1)) # NOTE: Need to squeeze the inner dimension, otherwise the optimization fails in acquisition_base.find_eta()

		# print("X.shape: ",X.shape)
		# print("posterior_X.mean.shape: ",posterior_X.mean.shape)

		return -posterior_X.mean # b-dimensional vector
