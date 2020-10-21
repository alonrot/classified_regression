import numpy as np
import torch
from classireg.models.gpmodel import GPmodel

class Simple1DSequentialGP():
    def __init__(self, gp: GPmodel):
        """
        Construct a virtual function, which evaluations are sampled from
        the current GP posterior, and sequentially included in the dataset
        """
        self.gp = gp
        self.x_gm = np.array([[0.0]]) # Dummy value
    def evaluate(self,x_in,with_noise=True):
        return self.gp(x_in).sample(torch.Size([1])).item()
    def true_minimum(self):
        return self.x_gm, -3.0 # Dummy value
    def __call__(self,x_in,with_noise=False):
        return torch.Tensor([self.evaluate(x_in,with_noise)])