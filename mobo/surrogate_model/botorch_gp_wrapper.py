from typing import List
import numpy as np
import torch

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

from botorch.models.gp_regression import (
    HeteroskedasticSingleTaskGP,
    SingleTaskGP,
)

from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import LikelihoodList
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model, fit_gpytorch_mll_torch
from botorch.models.transforms.outcome import Standardize

from mobo.surrogate_model.base import SurrogateModel

import gpytorch
import torch

from .botorch_helper import ZeroKernel, CustomHeteroskedasticSingleTaskGP

class BoTorchSurrogateModel(SurrogateModel):
    """
    Gaussian process
    """

    def __init__(self, n_var, n_obj, **kwargs):
        self.bo_model = None
        self.mll = None
        self.state_dict = None
        self.input_transform = None
        super().__init__(n_var, n_obj)

    def _fit(self, X_torch, Y_torch, rho_torch=None):

        try:
            mll, self.bo_model = self.initialize_model(
                X_torch.clone(), Y_torch.clone(), rho_torch.clone()
            )
            fit_gpytorch_mll(mll, max_retries=5)
            return
        except RuntimeError as e:
            print(e)
            print("failed to fit, retrying with torch optim...")

        try:
            mll, self.bo_model = self.initialize_model(
                X_torch.clone(), Y_torch.clone(), rho_torch.clone()
            )
            fit_gpytorch_mll_torch(mll, step_limit=2000)
        except RuntimeError as e:
            print(e)
            print("failed to fit. Keeping the previous model.")

    def fit(self, X, Y, rho=None):
        X_torch = torch.tensor(X).to(**tkwargs).detach()
        Y_torch = torch.tensor(Y).to(**tkwargs).detach()
        rho_torch = (
            torch.tensor(rho).to(**tkwargs).detach() if rho is not None else None
        )
        print("rho_max", rho_torch.max())

        mll, self.bo_model = self.initialize_model(
            X_torch.clone(), Y_torch.clone(), rho_torch.clone()
        )
        self._fit(X_torch, Y_torch, rho_torch)
        # print state dict except input_transform
        for i, m in enumerate(self.bo_model.models):
            for p in m.named_parameters():
                print(p)
            
    def initialize_model(self, train_x, train_y, train_rho=None):
        # define models for objective and constraint
        train_y_mean = -train_y  # negative because botorch assumes maximization
        train_y_var = train_rho + 1e-6
        
        models = []
        for i in range(self.n_obj):
            model = CustomHeteroskedasticSingleTaskGP(
                train_X=train_x,
                train_Y=train_y_mean[..., i:i+1],
                train_Yvar=train_y_var[..., i:i+1],
                input_transform=self.input_transform,
                outcome_transform=Standardize(m=1),
            )            
            
            models.append(model)
        
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        
        if self.state_dict is not None:
            try:
                #replace each input_transform with the one from the new model
                for i, m in enumerate(model.models):
                    self.bo_model.models[i].input_transform = m.input_transform 
                self.state_dict = self.bo_model.state_dict()
                
                model.load_state_dict(self.state_dict, strict=False)
            except Exception as e:
                print(e)
                print("failed to load state dict")

        return mll, model

    def evaluate(
        self,
        X,
        std=False,
        noise=False,
        calc_gradient=False,
        calc_hessian=False,
        calc_mvar=False,
    ):
        X = torch.tensor(X).to(**tkwargs)

        F, dF, hF = None, None, None  # mean
        S, dS, hS = None, None, None  # std
        rho_F, drho_F = None, None  # noise mean
        rho_S, drho_S = None, None  # noise std

        model = self.bo_model
        
        post = self.bo_model.posterior(X)
        # negative because botorch assumes maximization (undo previous negative)
        F = -post.mean.squeeze(-1).detach().cpu().numpy()
        S = post.variance.sqrt().squeeze(-1).detach().cpu().numpy()

        if noise:
            if isinstance(model.likelihood, LikelihoodList):
                rho_F = np.zeros_like(F)
                rho_S = np.zeros_like(S)
                for i, likelihood in enumerate(model.likelihood.likelihoods):
                    if hasattr(likelihood.noise_covar, "noise_model"):
                        rho_post = likelihood.noise_covar.noise_model.posterior(X)
                        rho_F[:, i] = rho_post.mean.detach().cpu().squeeze(-1).numpy()
                        rho_S[:, i] = rho_post.variance.sqrt().detach().cpu().squeeze(-1).numpy()
            else:
                rho_post = model.likelihood.noise_covar.noise_model.posterior(X)
                rho_F = rho_post.mean.detach().cpu().numpy()
                rho_S = rho_post.variance.sqrt().detach().cpu().numpy()

        # #simplest 2d --> 2d test problem
        # def f(X):
        #     return torch.stack([X[:, 0]**2 + 0.1*X[:, 1]**2 , -X[:, 1]**2 -0.1*(X[:, 0]**2)]).T
        # X_toy = torch.tensor([[0.5, 0.5], [1., 1.], [2., 2.]], requires_grad=True)
        # jacobian_mean = torch.autograd.functional.jacobian(f, X_toy)
        # # goal 3 x 2 x 2
        # jac_batch = jacobian_mean.diagonal(dim1=0,dim2=2).transpose(0,-1).transpose(1,2).numpy()

        if calc_gradient:
            jac_F = torch.autograd.functional.jacobian(
                lambda x: -self.bo_model(x).mean.T, X
            )
            dF = (
                jac_F.diagonal(dim1=0, dim2=2)
                .transpose(0, -1)
                .transpose(1, 2)
                .detach()
                .cpu()
                .numpy()
            )

            if std:
                jac_S = torch.autograd.functional.jacobian(
                    lambda x: self.bo_model(x).variance.sqrt().T, X
                )
                dS = (
                    jac_S.diagonal(dim1=0, dim2=2)
                    .transpose(0, -1)
                    .transpose(1, 2)
                    .detach()
                    .cpu()
                    .numpy()
                )

        out = {
            "F": F,
            "dF": dF,
            "hF": hF,
            "S": S,
            "dS": dS,
            "hS": hS,
            "rho_F": rho_F,
            "drho_F": drho_F,
            "rho_S": rho_S,
            "drho_S": drho_S,
            "mvar_F": np.zeros_like(F),
        }

        return out

class BoTorchSurrogateModelMean(BoTorchSurrogateModel):

    # last objective is the "deterministic" one
    
    def initialize_model(self, train_x, train_y, train_rho=None):
        # define models for objective and constraint
        train_y_mean = -train_y  # negative because botorch assumes maximization
        train_y_var = train_rho + 1e-10
        
        models = []
        for i in range(self.n_obj-1):
            model = CustomHeteroskedasticSingleTaskGP(
                train_X=train_x,
                train_Y=train_y_mean[..., i:i+1],
                train_Yvar=train_y_var[..., i:i+1],
                input_transform=self.input_transform,
                outcome_transform=Standardize(m=1),
            )            
            
            models.append(model)
        
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        class LinearMean(gpytorch.means.Mean):
            def __init__(self):        
                super(LinearMean, self).__init__()
                
            def forward(self, x):
                """Your forward method."""
                return x[..., 1]
            
        model_mean = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y_mean[..., -1:],
            train_Yvar=train_y_var[..., -1:]*0.0,
            likelihood= gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-10)),
            input_transform=self.input_transform,
            # outcome_transform=Standardize(m=1),
            mean_module=LinearMean(),
            covar_module=ZeroKernel(),
        )
        model_mean.likelihood.noise_covar.noise = 1e-10
        
        models.append(model_mean)
        model = ModelListGP(*models)
        
        if self.state_dict is not None:
            try:
                #replace each input_transform with the one from the new model
                for i, m in enumerate(model.models):
                    self.bo_model.models[i].input_transform = m.input_transform 
                self.state_dict = self.bo_model.state_dict()
                
                model.load_state_dict(self.state_dict, strict=False)
            except Exception as e:
                print(e)
                print("failed to load state dict")

        return mll, model
