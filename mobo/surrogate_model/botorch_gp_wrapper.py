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
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model, fit_gpytorch_mll_torch

import botorch

from mobo.surrogate_model.base import SurrogateModel
from mobo.utils import safe_divide

from linear_operator.settings import _fast_solves

class BoTorchSurrogateModel(SurrogateModel):

    """
    Gaussian process
    """

    def __init__(self, n_var, n_obj, **kwargs):
        self.bo_model = None
        self.mll = None
        super().__init__(n_var, n_obj)

    def fit(self, X, Y, rho=None):
        X_torch = torch.tensor(X).to(**tkwargs).detach()
        Y_torch = torch.tensor(Y).to(**tkwargs).detach()
        rho_torch = torch.tensor(rho).to(**tkwargs).detach() if rho is not None else None
        mll, self.bo_model = self.initialize_model(X_torch, Y_torch, rho_torch)
        fit_gpytorch_model(mll)
        # fit_gpytorch_mll_torch(mll, step_limit=1000)
        

    def initialize_model(self, train_x, train_y, train_rho=None):
        # define models for objective and constraint
        train_y_mean = -train_y  # negative because botorch assumes maximization
        # train_y_var = self.real_problem.evaluate(train_x).to(**tkwargs).var(dim=-1)
        train_y_var = train_rho + 1e-6
        models = []
        for i in range(train_y.shape[1]):
            train_y_i = train_y_mean[..., i]
            train_yvar_i = train_y_var[..., i]
            # models.append(
            #     FixedNoiseGP(
            #         train_X=train_x,
            #         train_Y=train_y_i.unsqueeze(-1),
            #         train_Yvar=train_yvar_i.unsqueeze(-1),
            #         outcome_transform=Standardize(m=1),
            #     )
            # )
            models.append(
                SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y_i.unsqueeze(-1),
                    train_Yvar=train_yvar_i.unsqueeze(-1),
                )
            )
            # models.append(
            #     SingleTaskGP(
            #         train_x,
            #         train_y_i.unsqueeze(-1),
            #         # train_yvar_i.unsqueeze(-1),
            #         outcome_transform=Standardize(m=1),
            #     )
            # )
        # model = ModelListGP(*models)
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y_mean,
            train_Yvar=train_y_var,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll, max_retries=5)

        return mll, model

    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        X = torch.tensor(X).to(**tkwargs)

        F, dF, hF = None, None, None  # mean
        S, dS, hS = None, None, None  # std

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
        # for idx, ll in enumerate(self.bo_model.likelihood.likelihoods):
        #     if hasattr(ll, "noise_covar"):
        #         try:
        #             rho[:, idx] = ll.noise_covar.noise_model.posterior(X).mean.squeeze(-1).detach().cpu().numpy()
        #         except:
        #             pass

        
        # #simplest 2d --> 2d test problem
        # def f(X):
        #     return torch.stack([X[:, 0]**2 + 0.1*X[:, 1]**2 , -X[:, 1]**2 -0.1*(X[:, 0]**2)]).T
        # X_toy = torch.tensor([[0.5, 0.5], [1., 1.], [2., 2.]], requires_grad=True)
        # jacobian_mean = torch.autograd.functional.jacobian(f, X_toy)
        # # goal 3 x 2 x 2
        # jac_batch = jacobian_mean.diagonal(dim1=0,dim2=2).transpose(0,-1).transpose(1,2).numpy()

        if calc_gradient:
            jac_F = torch.autograd.functional.jacobian(lambda x: -self.bo_model(x).mean.T, X)
            dF = jac_F.diagonal(dim1=0,dim2=2).transpose(0,-1).transpose(1,2).detach().numpy()

        if std and calc_gradient:
            jac_S = torch.autograd.functional.jacobian(lambda x: self.bo_model(x).variance.sqrt().T, X)
            dS = jac_S.diagonal(dim1=0,dim2=2).transpose(0,-1).transpose(1,2).detach().numpy()
        
        out = {"F": F, "dF": dF, "hF": hF, "S": S, "dS": dS, "hS": hS, "rho": rho}
        return out
