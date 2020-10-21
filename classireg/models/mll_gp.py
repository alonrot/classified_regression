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

from torch.distributions.beta import Beta
from torch.distributions.gamma import Gamma

class MLLGP():

    def __init__(self, model_gp, likelihood_gp, hyperpriors: dict) -> None:
        self.model_gp = model_gp
        self.likelihood_gp = likelihood_gp
        self.hyperpriors = hyperpriors

        a_beta = self.hyperpriors["lengthscales"].kwds["a"]
        b_beta = self.hyperpriors["lengthscales"].kwds["b"]

        self.Beta_tmp = Beta(concentration1=a_beta, concentration0=b_beta)

        a_gg = self.hyperpriors["outputscale"].kwds["a"]
        b_gg = self.hyperpriors["outputscale"].kwds["scale"]

        self.Gamma_tmp = Gamma(concentration=a_gg,rate=1./b_gg)


    def log_marginal(self, lengthscales, outputscale) -> float:
        """
        """

        # print("lengthscales.shape:",lengthscales.shape)
        # print("outputscale.shape:",outputscale.shape)
        if lengthscales.dim() == 3 or outputscale.dim() == 3:
            Nels = lengthscales.shape[0]
            loss_vec = torch.zeros(Nels)
            for k in range(Nels):
                loss_vec[k] = self.log_marginal(lengthscales[k,0,:],outputscale[k,0,:])
            return loss_vec

        assert lengthscales.dim() <= 1 and outputscale.dim() <= 1

        assert not torch.any(torch.isnan(lengthscales)) and not torch.any(torch.isinf(lengthscales)), "lengthscales is inf or NaN"
        assert not torch.isnan(outputscale) and not torch.isinf(outputscale), "outputscale is inf or NaN"

        # Update hyperparameters:
        self.model_gp.covar_module.outputscale = outputscale
        self.model_gp.covar_module.base_kernel.lengthscale = lengthscales

        # self.model_gp.display_hyperparameters()

        # Get the log prob of the marginal distribution:
        function_dist = self.model_gp(self.model_gp.train_inputs[0])
        output = self.likelihood_gp(function_dist)
        loss_val = output.log_prob(self.model_gp.train_targets).view(1)

        # if self.debug == True:
        #     pdb.set_trace()

        loss_lengthscales_hyperprior = torch.sum(self.Beta_tmp.log_prob(lengthscales)).view(1)
        loss_outputscale_hyperprior = self.Gamma_tmp.log_prob(outputscale)

        # loss_lengthscales_hyperprior = sum(self.hyperpriors["lengthscales"].logpdf(lengthscales))
        # loss_outputscale_hyperprior = self.hyperpriors["outputscale"].logpdf(outputscale).item()

        loss_val += loss_lengthscales_hyperprior + loss_outputscale_hyperprior

        try:
            assert not torch.any(torch.isnan(loss_val)) and not torch.any(torch.isinf(loss_val)), "loss_val is Inf or NaN"
        except: # debug TODO DEBUG
            logger.info("loss_val: {0:s}".format(str(loss_val)))
            logger.info("loss_lengthscales_hyperprior: {0:s}".format(str(loss_lengthscales_hyperprior)))
            logger.info("loss_outputscale_hyperprior: {0:s}".format(str(loss_outputscale_hyperprior)))

        return loss_val

    def __call__(self, pars_in):
        # Slice only last dimension: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.narrow
        lengthscales = pars_in.narrow(  dim=-1,
                                        start=self.model_gp.idx_hyperpars["lengthscales"][0],
                                        length=len(self.model_gp.idx_hyperpars["lengthscales"]))


        outputscale = pars_in.narrow(   dim=-1,
                                        start=self.model_gp.idx_hyperpars["outputscale"][0],
                                        length=len(self.model_gp.idx_hyperpars["outputscale"]))

        return -self.log_marginal(lengthscales,outputscale) # Use minus (-) when minizing the marginal likelihood



