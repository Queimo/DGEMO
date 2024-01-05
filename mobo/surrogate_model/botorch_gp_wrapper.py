import numpy as np
import torch

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

from botorch.models.gp_regression import FixedNoiseGP, HeteroskedasticSingleTaskGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

from mobo.surrogate_model.base import SurrogateModel
from mobo.utils import safe_divide


class BoTorchSurrogateModel(SurrogateModel):
    
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    
    '''
    Gaussian process
    '''
    def __init__(self, n_var, n_obj, **kwargs):
        self.bo_model = None
        self.mll = None
        super().__init__(n_var, n_obj)
        

    def fit(self, X, Y):
        X_torch = torch.from_numpy(X)
        Y_torch = torch.from_numpy(Y)
        self.mll, self.bo_model = self.initialize_model(X_torch, Y_torch)
        fit_gpytorch_mll(self.mll)
        
    def initialize_model(self, train_x, train_y):
        # define models for objective and constraint
        train_y_mean = train_y
        # train_y_var = self.real_problem.evaluate(train_x).to(**tkwargs).var(dim=-1)
        train_y_var = torch.zeros_like(train_y).to(**tkwargs)
        models = []
        for i in range(train_y.shape[1]):
            train_y_i = train_y_mean[..., i]
            train_yvar_i = train_y_var[..., i]
            models.append(
                FixedNoiseGP(
                    train_x,
                    train_y_i.unsqueeze(-1),
                    train_yvar_i.unsqueeze(-1),
                    outcome_transform=Standardize(m=1),
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

        
    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        
        X = torch.from_numpy(X)
        
        F, dF, hF = [], [], [] # mean
        S, dS, hS = [], [], [] # std
        
        F = self.bo_model.posterior(X).mean.squeeze(-1).detach().cpu().numpy()
        S = self.bo_model.posterior(X).variance.squeeze(-1).T.detach().cpu().numpy()
        
        dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None
        
        S = np.stack(S, axis=1) if std else None
        dS = np.stack(dS, axis=1) if std and calc_gradient else None
        hS = np.stack(hS, axis=1) if std and calc_hessian else None

        out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
        return out
