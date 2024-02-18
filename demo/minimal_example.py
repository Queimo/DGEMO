import torch
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective

import os
from botorch.models.gp_regression import (
    HeteroskedasticSingleTaskGP,
)

from botorch.utils.transforms import (
    concatenate_pending_points,
    is_ensemble,
    match_batch_shape,
    t_batch_mode_transform,
    standardize
)
from botorch.utils.safe_math import (
    fatmin,
    log_fatplus,
    log_softplus,
    logdiffexp,
    logmeanexp,
    logplusexp,
    logsumexp,
    smooth_amin,
)

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize

SMOKE_TEST = os.environ.get("SMOKE_TEST")
from botorch.sampling.normal import SobolQMCNormalSampler
from ref_point import RefPoint

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

class qLogNEHVI(qNoisyExpectedHypervolumeImprovement):
    
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X): 
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # NOTE: To ensure that we correctly sample `f(X)` from the joint distribution
        # `f((X_baseline, X)) ~ P(f | D)`, it is critical to compute the joint posterior
        # over X *and* X_baseline -- which also contains pending points whenever there
        # are any --  since the baseline and pending values `f(X_baseline)` are
        # generally pre-computed and cached before the `forward` call, see the docs of
        # `cache_pending` for details.
        # TODO: Improve the efficiency by not re-computing the X_baseline-X_baseline
        # covariance matrix, but only the covariance of
        # 1) X and X, and
        # 2) X and X_baseline.
        posterior = self.model.posterior(X_full)
        # Account for possible one-to-many transform and the MCMC batch dimension in
        # `SaasFullyBayesianSingleTaskGP`
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

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
MC_SAMPLES = 128 if not SMOKE_TEST else 16

batch_size = 1
standard_bounds = torch.zeros(2, 2, **tkwargs)
standard_bounds[1] = 1

from problems.peaks import Peaks

# d=2
# m=2
problem = Peaks(sigma=0.7, repeat_eval=3)
dim = problem.dim
ref_point_handler = RefPoint(
    "peaks", problem.dim, problem.num_objectives, n_init_sample=100, seed=0
)

ref_point = ref_point_handler.get_ref_point(is_botorch=True)


def evaluate_function(X):
    # Shape X (b, d)
    # Shape Y (b, m, n_w)
    Y = problem.evaluate_repeat(unnormalize(X, problem.bounds).numpy())
    Y = torch.tensor(Y).to(**tkwargs).mean(dim=-1)
    return Y


def train_model(train_X, train_Y):
    # define models for objective and constraint
    model = HeteroskedasticSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=torch.ones_like(train_Y),
        outcome_transform=Standardize(m=problem.num_objectives),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, max_retries=5)
    return mll, model


sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

X_init = torch.rand(12, 2, **tkwargs)
Y_init = evaluate_function(X_init)  # shape (10, 2, 3) (3 repeats)

model = train_model(X_init, Y_init)

# objective = ...  # Simple mean-variance objective
class MeanVarianceObjective(MCMultiOutputObjective):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, samples):
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean - self.beta * std

beta = 0.5  # Adjust this based on your risk preference
objective = MeanVarianceObjective(beta)

def solve(X, Y):
    acq_func = qLogNEHVI(
        model=model,
        ref_point=ref_point,
        # objective=objective,
        X_baseline=X,
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )

    X_cand, Y_cand_pred = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=batch_size,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": batch_size, "maxiter": 2000},
        sequential=True,
    )

    return X_cand, Y_cand_pred

solve(X_init, Y_init)

# BO Loop
...
