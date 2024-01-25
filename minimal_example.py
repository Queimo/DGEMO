import torch
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

import os
from botorch.models.gp_regression import (
    HeteroskedasticSingleTaskGP,
)


from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.fit import fit_gpytorch_mll

SMOKE_TEST = os.environ.get("SMOKE_TEST")
from botorch.sampling.normal import SobolQMCNormalSampler
from ref_point import RefPoint

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
MC_SAMPLES = 128 if not SMOKE_TEST else 16

batch_size = 1
standard_bounds = torch.zeros(2, 2, **tkwargs)
standard_bounds[1] = 1

from problems.k1 import K5

# d=2
# m=2
problem = K5(sigma=0.7, repeat_eval=3)
dim = problem.dim
ref_point_handler = RefPoint(
    "k5", problem.dim, problem.num_objectives, n_init_sample=100, seed=0
)

ref_point = ref_point_handler.get_ref_point(is_botorch=True)


def evaluate_function(X):
    # Shape X (b, d)
    # Shape Y (b, m, n_w)
    Y = problem.evaluate_repeat(unnormalize(X, problem.bounds).numpy())
    Y = torch.tensor(Y).to(**tkwargs)
    return Y


def train_model(train_X, train_Y):
    # define models for objective and constraint
    model = HeteroskedasticSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=...,  # AppendFeatures??
        outcome_transform=...,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, max_retries=5)
    return mll, model


sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

X_init = torch.rand(10, 2, **tkwargs)
Y_init = evaluate_function(X_init)  # shape (10, 2, 3) (3 repeats)

model = train_model(X_init, Y_init)

objective = ...  # Simple mean-variance objective


def solve(X, Y):
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        objective=objective,
        X_baseline=torch.from_numpy(X).to(**tkwargs),
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


# BO Loop
...
