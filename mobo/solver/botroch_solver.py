from . import NSGA2Solver, Solver
from mobo.solver.mvar_edit import MVaR, get_MARS, get_nehvi_ref_point
from pymoo.algorithms.nsga2 import NSGA2
from abc import abstractmethod

import numpy as np
import torch

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
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
import os
import gc

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
MC_SAMPLES = 128 if not SMOKE_TEST else 16


class DeterministicModel3(DeterministicModel):

    def __init__(self, num_outputs, input_transform=None, **kwargs):
        super().__init__()
        self.input_transform = input_transform

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = X[..., 1].unsqueeze(-1)
        return y


model3 = DeterministicModel3(
    num_outputs=1,
    input_transform=InputPerturbation(torch.zeros((11, 2), **tkwargs)),
)


class BoTorchSolver(NSGA2Solver):
    """
    Solver based on PSL
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def bo_solve(self, problem, X, Y, rho):
        pass

    def nsga2_solve(self, problem, X, Y):
        return super().solve(problem, X, Y)


    def optimize_acqf_loop(self, problem, acq_func):
            
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
                    sequential=True,
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


class qLogNEHVI(qNoisyExpectedHypervolumeImprovement):
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
        # Add previous nehvi from pending points.
        return self._compute_qehvi(samples=samples, X=X) + self._prev_nehvi



def get_nehvi(
    model,
    X_baseline,
    sampler,
    ref_point,
    ref_aware: bool = False,
    objective=None,
):
    r"""Construct the NEHVI acquisition function.

    Args:
        model: A fitted multi-output GPyTorchModel.
        X_baseline: An `r x d`-dim tensor of points already observed.
        sampler: The sampler used to draw the base samples.
        num_constraints: The number of constraints. Constraints are handled by
            weighting the samples according to a sigmoid approximation of feasibility.
            If objective is given, it is applied after weighting.
        use_rff: If True, replace the model with a single RFF sample. NEHVI-1.
        ref_point: The reference point.
        ref_aware: If True, use the given ref point. Otherwise, approximate it.
        num_rff_features: The number of RFF features.
        objective: This is the optional objective for optimizing independent
            risk measures, such as expectation.

    Returns:
        The qNEHVI acquisition function.
    """
    if ref_point is None or not ref_aware:
        ref_point = get_nehvi_ref_point(
            model=model, X_baseline=X_baseline, objective=objective
        )
    return qLogNEHVI(
        model=model,
        ref_point=ref_point,
        X_baseline=X_baseline,
        sampler=sampler,
        objective=objective,
        prune_baseline=True,
    )


class RAqNEHVISolver(BoTorchSolver):
    """
    Solver based on PSL
    """

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs["alpha"]
        print("alpha", self.alpha)

        super().__init__(*args, **kwargs)

    def bo_solve(self, problem, X, Y, rho):
        surrogate_model = problem.surrogate_model
        
        model = surrogate_model.bo_model
        X_baseline=torch.from_numpy(X).to(**tkwargs)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        objective = MVaR(n_w=11, alpha=self.alpha)

        ref_point = get_nehvi_ref_point(
                model=model, X_baseline=X_baseline, objective=objective
            )
        
        acq_func = qLogNEHVI(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            prune_baseline=True,
        )
        
        selection = self.optimize_acqf_loop(self, problem, acq_func)

        return selection


class qNEI(qNoisyExpectedImprovement):

    def __init__(
        self,
        model,
        X_baseline,
        objective,
        prune_baseline,
        sampler,
        det_model,
    ):
        super().__init__(
            model=model,
            X_baseline=X_baseline,
            objective=objective,
            prune_baseline=prune_baseline,
            sampler=sampler,
        )

        self.det_model = det_model
        self.obj3 = objective

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
        self._cache_root = False
        if not self._cache_root:
            samples_full = super().get_posterior_samples(posterior)

            y3 = self.det_model.posterior(X_full).mean
            y3_sa = y3.expand([MC_SAMPLES, -1, -1, -1])
            # concatenate the samples_full and y3
            samples_full = torch.cat([samples_full, y3_sa], dim=-1)

            samples = samples_full[..., -q:, :]
            obj_full = self.obj3(samples_full, X=X_full)
            # assigning baseline buffers so `best_f` can be computed in _sample_forward
            self.baseline_obj, obj = obj_full[..., :-q], obj_full[..., -q:]
            self.baseline_samples = samples_full[..., :-q, :]
        # else:
        #     # handle one-to-many input transforms
        #     n_plus_q = X_full.shape[-2]
        #     n_w = posterior._extended_shape()[-2] // n_plus_q
        #     q_in = q * n_w
        #     self._set_sampler(q_in=q_in, posterior=posterior)
        #     samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        #     obj = self.objective(samples, X=X_full[..., -q:, :])

        return samples, obj


class MARSSolver(BoTorchSolver):
    """
    Solver based on PSL
    """

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs["alpha"]
        print("alpha", self.alpha)

        super().__init__(*args, **kwargs)

    def bo_solve(self, problem, X, Y, rho):
        surrogate_model = problem.surrogate_model

        X_baseline = torch.from_numpy(X).to(**tkwargs)
        model = surrogate_model.bo_model

        Y_12 = model.posterior(X_baseline, observation_noise=True).mean
        Y_3 = model3.posterior(X_baseline).mean
        Y_samples = torch.cat([Y_12, Y_3], dim=1)
        Y_samples = model.posterior(X_baseline, observation_noise=False).mean
        # Y_samples = None

        mvar_obj = MVaR(n_w=11, alpha=self.alpha)
        ref_point = get_nehvi_ref_point(
            model=model,
            X_baseline=torch.from_numpy(X).to(**tkwargs),
            objective=mvar_obj,
        )
        print("mvar_ref_point", ref_point)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        mars = get_MARS(
            model=model,
            n_w=11,
            X_baseline=X_baseline,
            sampler=sampler,
            mvar_ref_point=ref_point,
            alpha=self.alpha,
            Y_samples=Y_samples,
        )
        
        acq_func = qNEI(
            model=model,
            X_baseline=X_baseline,
            objective=mars,
            prune_baseline=False,
            sampler=sampler,
            det_model=model3,
        )
        
        selection = self.optimize_acqf_loop(self, problem, acq_func)
        return selection


class qNEHVISolver(BoTorchSolver):
    """
    Solver based on PSL
    """

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
        
        selection = self.optimize_acqf_loop(self, problem, acq_func)

        return selection


class qEHVISolver(BoTorchSolver):
    """
    Solver based on PSL
    """

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
        
        selection = self.optimize_acqf_loop(self, problem, acq_func)
        
        return selection
