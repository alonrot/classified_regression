#!/usr/bin/env python3
import torch
from gpytorch.distributions.multivariate_normal import MultivariateNormal
# from ..likelihoods import _GaussianLikelihoodBase
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from classireg.utils.parsing import get_logger
import numpy as np
import pdb
logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class MLLGPCR():

    def __init__(self, model_gpcr, hyperpriors: dict) -> None:
        self.model_gpcr = model_gpcr
        self.hyperpriors = hyperpriors

    def log_marginal(self, lengthscales, outputscale, threshold) -> float:
        """
        """

        assert not torch.any(torch.isnan(lengthscales)) and not torch.any(torch.isinf(lengthscales)), "lengthscales is inf or NaN"
        assert not torch.isnan(outputscale) and not torch.isinf(outputscale), "outputscale is inf or NaN"
        assert not torch.isnan(threshold) and not torch.isinf(threshold), "threshold is inf or NaN"

        # Update hyperparameters:
        self.model_gpcr.covar_module.outputscale = outputscale
        self.model_gpcr.covar_module.base_kernel.lengthscale = lengthscales
        self.model_gpcr.threshold = threshold

        # self.model_gpcr.display_hyperparameters()

        # Update EP posterior to be able to acquire the logZ:
        self.model_gpcr._update_prior()
        self.model_gpcr._update_EP_object()
        self.model_gpcr._update_approximate_posterior()

        # Use by default MLL type I:
        loss_val = self.model_gpcr.logZ

        # Add MLL type II:
        loss_lengthscales_hyperprior = sum(self.hyperpriors["lengthscales"].logpdf(lengthscales))
        loss_lengthscales_outputscale = self.hyperpriors["outputscale"].logpdf(outputscale).item()
        loss_lengthscales_thres = self.hyperpriors["threshold"].logpdf(threshold).item()
        loss_val += loss_lengthscales_hyperprior + loss_lengthscales_outputscale + loss_lengthscales_thres

        try:
            assert not np.any(np.isnan(loss_val)) and not np.any(np.isinf(loss_val)), "loss_val is Inf or NaN"
        except: # debug TODO DEBUG
            logger.info("loss_val: {0:s}".format(str(loss_val)))
            logger.info("loss_lengthscales_hyperprior: {0:s}".format(str(loss_lengthscales_hyperprior)))
            logger.info("loss_lengthscales_outputscale: {0:s}".format(str(loss_lengthscales_outputscale)))
            logger.info("loss_lengthscales_thres: {0:s}".format(str(loss_lengthscales_thres)))

        return loss_val

    def __call__(self, pars_in):
        return -self.log_marginal(  pars_in[self.model_gpcr.idx_hyperpars["lengthscales"]],
                                    pars_in[self.model_gpcr.idx_hyperpars["outputscale"]],
                                    pars_in[self.model_gpcr.idx_hyperpars["threshold"]])



