import numpy as np
import torch

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

from botorch.models.gp_regression import (
    FixedNoiseGP,
    HeteroskedasticSingleTaskGP,
    SingleTaskGP,
)
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_scipy

import botorch

from mobo.surrogate_model.base import SurrogateModel
from mobo.utils import safe_divide


class BoTorchSurrogateModel(SurrogateModel):

    """
    Gaussian process
    """

    def __init__(self, n_var, n_obj, **kwargs):
        self.bo_model = None
        self.mll = None
        super().__init__(n_var, n_obj)

    def fit(self, X, Y, rho=None):
        X_torch = torch.tensor(X).to(**tkwargs)
        Y_torch = torch.tensor(Y).to(**tkwargs)
        rho_torch = torch.tensor(rho).to(**tkwargs) if rho is not None else None
        self.mll, self.bo_model = self.initialize_model(X_torch, Y_torch, rho_torch)
        fit_gpytorch_scipy(self.mll)

    def initialize_model(self, train_x, train_y, train_rho=None):
        # define models for objective and constraint
        train_y_mean = -train_y  # negative because botorch assumes maximization
        # train_y_var = self.real_problem.evaluate(train_x).to(**tkwargs).var(dim=-1)
        train_y_var = train_rho
        models = []
        for i in range(train_y.shape[1]):
            train_y_i = train_y_mean[..., i]
            train_yvar_i = train_y_var[..., i]
            models.append(
                FixedNoiseGP(
                    train_X=train_x,
                    train_Y=train_y_i.unsqueeze(-1),
                    train_Yvar=train_yvar_i.unsqueeze(-1)*0. + 1e-6,
                )
            )
            # models.append(
            #     HeteroskedasticSingleTaskGP(
            #         train_X=train_x,
            #         train_Y=train_y_i.unsqueeze(-1),
            #         train_Yvar=train_yvar_i.unsqueeze(-1),
            #         outcome_transform=Standardize(m=1),
            #     )
            # )
            # models.append(
            #     SingleTaskGP(
            #         train_x,
            #         train_y_i.unsqueeze(-1),
            #         # train_yvar_i.unsqueeze(-1),
            #         outcome_transform=Standardize(m=1),
            #     )
            # )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        return mll, model

    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        X = torch.tensor(X, requires_grad=True).to(**tkwargs)

        F, dF, hF = [], [], []  # mean
        S, dS, hS = [], [], []  # std

        F = (
            -self.bo_model.posterior(X).mean.squeeze(-1).detach().cpu().numpy()
        )  # negative because botorch assumes maximization (undo previous negative)
        
        
        S = (
            self.bo_model.posterior(X)
            .variance.sqrt()
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        
        rho = np.zeros((X.shape[0], self.n_obj))
        for idx, ll in enumerate(self.bo_model.likelihood.likelihoods):
            if hasattr(ll, "noise_covar"):
                try:
                    rho[:, idx] = ll.noise_covar.noise_model.posterior(X).mean.squeeze(-1).detach().cpu().numpy()
                except:
                    pass

        # dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None
        
        
        S = np.stack(S, axis=1) if std else None

        S = np.stack(S, axis=1) if std else None
        
        for model in self.bo_model.models:
            if calc_gradient:
                # Compute the Jacobian for the mean
                jacobian_mean = torch.autograd.functional.jacobian(lambda x: -model.posterior(x).mean, X).squeeze(1)
                # Extract the diagonal elements and reshape to [10, 2, 1]
                dF.append(jacobian_mean.diagonal(dim1=0, dim2=1).detach().T.numpy())
            else:
                dF.append(None)

            if std and calc_gradient:
                # Compute the Jacobian for the standard deviation
                jacobian_std = torch.autograd.functional.jacobian(lambda x: model.posterior(x).variance.sqrt(), X).squeeze(1)
                # Extract the diagonal elements and reshape to [10, 2, 1]
                dS.append(jacobian_std.diagonal(dim1=0, dim2=1).detach().T.numpy())
            else:
                dS.append(None)

        dF = np.stack(dF, axis=1) if calc_gradient else None
        dS = np.stack(dS, axis=1) if std and calc_gradient else None
        
        # dS = np.stack(dS, axis=1) if std and calc_gradient else None
        hS = np.stack(hS, axis=1) if std and calc_hessian else None

        out = {"F": F, "dF": dF, "hF": hF, "S": S, "dS": dS, "hS": hS, "rho": rho}
        return out
