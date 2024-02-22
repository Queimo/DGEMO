from . import NSGA2Solver, Solver
from mobo.solver.mvar_edit import MVaR, get_MARS, get_nehvi_ref_point
from pymoo.algorithms.nsga2 import NSGA2
from abc import abstractmethod

import numpy as np
import torch
import botorch

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
    logplusexp,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_ensemble,
    match_batch_shape,
    t_batch_mode_transform,
    standardize,
)
from botorch.models.transforms.input import InputPerturbation
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import ModelList
import os
import gc

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
print("SMOKE_TEST", SMOKE_TEST)
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
MC_SAMPLES = 128 if not SMOKE_TEST else 16


class BoTorchSolver(NSGA2Solver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = kwargs["alpha"]
        self.n_w = kwargs["n_w"]
        print("alpha", self.alpha)

    @abstractmethod
    def bo_solve(self, problem, X, Y, rho):
        pass

    def nsga2_solve(self, problem, X, Y):
        return super().solve(problem, X, Y)

    def optimize_acqf_loop(self, problem, acq_func, sequential=False):

        standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
        standard_bounds[1] = 1
        options = {"batch_limit": self.batch_size, "maxiter": 2000}

        while options["batch_limit"] >= 1:
            try:
                torch.cuda.empty_cache()
                X_cand, Y_cand_pred = optimize_acqf(
                    acq_function=acq_func,
                    bounds=standard_bounds,
                    q=self.batch_size,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                    options=options,
                    sequential=sequential,
                )
                torch.cuda.empty_cache()
                gc.collect()
                break
            except RuntimeError as e:
                if options["batch_limit"] > 1:
                    print(
                        "Got a RuntimeError in `optimize_acqf`. "
                        "Trying with reduced `batch_limit`."
                    )
                    options["batch_limit"] //= 2
                    continue
                else:
                    raise e

        selection = {
            "x": np.array(X_cand.detach().cpu()),
            "y": np.array(Y_cand_pred.detach().cpu()),
        }

        return selection

    def solve(self, problem, X, Y, rho):
        # get pareto_front with NSGA because botorch is slow for large batches
        self.solution = self.nsga2_solve(problem, X, Y)
        return self.bo_solve(problem, X, Y, rho)

class qNEHVI(qNoisyExpectedHypervolumeImprovement):
    
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        posterior = self.model.posterior(X_full, observation_noise=True)
        event_shape_lag = 1 if is_ensemble(self.model) else 2
        n_w = (
            posterior._extended_shape()[X_full.dim() - event_shape_lag]
            // X_full.shape[-2]
        )
        q_in = X.shape[-2] * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)

        return self._compute_qehvi(samples=samples, X=X) + self._prev_nehvi


class qLogNEHVI(qLogNoisyExpectedHypervolumeImprovement):
    
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        posterior = self.model.posterior(X_full, observation_noise=True)
        # Account for possible one-to-many transform and the model batch dimensions in
        # ensemble models.
        event_shape_lag = 1 if is_ensemble(self.model) else 2
        n_w = (
            posterior._extended_shape()[X_full.dim() - event_shape_lag]
            // X_full.shape[-2]
        )
        q_in = X.shape[-2] * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        # Add previous nehvi from pending points.
        nehvi = self._compute_log_qehvi(samples=samples, X=X)
        if self.incremental_nehvi:
            return nehvi
        return logplusexp(nehvi, self._prev_nehvi.log())


class RAqNEHVISolver(BoTorchSolver):

    def __init__(self, *args, **kwargs):
        self.sequential = True
        self.hvi_class = qNEHVI

        super().__init__(*args, **kwargs)

    def bo_solve(self, problem, X, Y, rho):
        surrogate_model = problem.surrogate_model

        model = surrogate_model.bo_model
        X_baseline = torch.from_numpy(X).to(**tkwargs)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        objective = MVaR(n_w=self.n_w, alpha=self.alpha)

        # model = ModelList(model, model3)

        ref_point = get_nehvi_ref_point(
            model=model, X_baseline=X_baseline, objective=objective
        )

        acq_func = self.hvi_class(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            prune_baseline=True,
            cache_root=True,
        )

        selection = self.optimize_acqf_loop(problem, acq_func, sequential=self.sequential)

        return selection
    
class RAqLogNEHVISolver(RAqNEHVISolver):
     
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hvi_class = qLogNEHVI


class qNEI(qNoisyExpectedImprovement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_samples_and_objectives(self, X):
        r"""Compute samples at new points, using the cached root decomposition.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A two-tuple `(samples, obj)`, where `samples` is a tensor of posterior
            samples with shape `sample_shape x batch_shape x q x m`, and `obj` is a
            tensor of MC objective values with shape `sample_shape x batch_shape x q`.
        """
        q = X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform, observation_noise=True
        )
        
        if not self._cache_root:
            samples_full = super().get_posterior_samples(posterior)
            samples = samples_full[..., -q:, :]
            obj_full = self.objective(samples_full, X=X_full)
            # assigning baseline buffers so `best_f` can be computed in _sample_forward
            self.baseline_obj, obj = obj_full[..., :-q], obj_full[..., -q:]
            self.baseline_samples = samples_full[..., :-q, :]
        else:
            # handle one-to-many input transforms
            n_plus_q = X_full.shape[-2]
            n_w = posterior._extended_shape()[-2] // n_plus_q
            q_in = q * n_w
            self._set_sampler(q_in=q_in, posterior=posterior)
            samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
            obj = self.objective(samples, X=X_full[..., -q:, :])

        return samples, obj


class MARSSolver(BoTorchSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def bo_solve(self, problem, X, Y, rho):
        surrogate_model = problem.surrogate_model

        X_baseline = torch.from_numpy(X).to(**tkwargs)
        model = surrogate_model.bo_model

        ref_point = get_nehvi_ref_point(
            model=model,
            X_baseline=X_baseline,
            objective=MVaR(n_w=self.n_w, alpha=self.alpha),
        )
        
        print("mvar_ref_point", ref_point)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        mars = get_MARS(
            model=model,
            n_obj=Y.shape[-1],
            n_w=self.n_w,
            X_baseline=X_baseline,
            alpha=self.alpha,
        )

        acq_func = qNEI(
            model=model,
            X_baseline=X_baseline,
            objective=mars,
            prune_baseline=True,
            sampler=sampler,
        )

        selection = self.optimize_acqf_loop(problem, acq_func)
        return selection


class qNEHVISolver(BoTorchSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def bo_solve(self, problem, X, Y, rho):
        surrogate_model = problem.surrogate_model

        ref_point = self.ref_point
        print("ref_point", ref_point)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        # solve surrogate problem
        # define acquisition functions
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            # acq_func = qLogNEHVI(
            model=surrogate_model.bo_model,
            ref_point=ref_point,  # use known reference point
            X_baseline=torch.from_numpy(X).to(**tkwargs),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )

        selection = self.optimize_acqf_loop(problem, acq_func)

        return selection


class qEHVISolver(BoTorchSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def bo_solve(self, problem, X, Y, rho):
        surrogate_model = problem.surrogate_model

        ref_point = self.ref_point
        print("ref_point", ref_point)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        # solve surrogate problem
        # define acquisition functions

        with torch.no_grad():
            pred = problem.evaluate(X)
        pred = torch.tensor(pred).to(**tkwargs)

        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(ref_point).to(**tkwargs),
            Y=pred,
        )
        acq_func = qLogExpectedHypervolumeImprovement(
            ref_point=torch.tensor(ref_point).to(**tkwargs),
            model=surrogate_model.bo_model,
            partitioning=partitioning,
            sampler=sampler,
        )

        selection = self.optimize_acqf_loop(problem, acq_func)

        return selection
