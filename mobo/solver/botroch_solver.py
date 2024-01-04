from . import Solver

import numpy as np
import torch

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

import os

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
import warnings

from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
MC_SAMPLES = 128 if not SMOKE_TEST else 16


class qNEHVISolver(Solver):
    '''
    Solver based on PSL
    '''
    def __init__(self, *args, **kwargs):
        
        super().__init__(algo="",*args,**kwargs)
        

    def solve(self, problem, X, Y):
        standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
        standard_bounds[1] = 1
        surrogate_model = problem.surrogate_model
        
        ref_point = self.ref_point
        print("ref_point", ref_point)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        # solve surrogate problem
        # define acquisition functions
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=surrogate_model.bo_model,
            ref_point=ref_point,  # use known reference point
            X_baseline=torch.from_numpy(X).to(**tkwargs),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        X_cand, Y_cand_pred = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.batch_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        
        
        # construct solution
        self.solution = {'x': np.array(X_cand), 'y': np.array(Y_cand_pred)}
        return self.solution


class qEHVISolver(Solver):
    '''
    Solver based on PSL
    '''
    def __init__(self, *args, **kwargs):
        
        super().__init__(algo="",*args,**kwargs)
        

    def solve(self, problem, X, Y):
        standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
        standard_bounds[1] = 1
        surrogate_model = problem.surrogate_model
        
        ref_point = torch.from_numpy(np.max(Y, axis=0)).to(**tkwargs)
        print("ref_point", ref_point)
        # ref_point =  torch.min(torch.cat((self.z.reshape(1,surrogate_model.n_obj),torch.from_numpy(Y).to(**tkwargs) - 0.1)), axis = 0).values.data
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        # solve surrogate problem
        # define acquisition functions
        with torch.no_grad():
            pred = problem.evaluate(X)
        pred = torch.from_numpy(pred).to(**tkwargs)
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=pred,
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=surrogate_model.bo_model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        # optimize
        X_cand, Y_cand_pred = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.batch_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        
        # construct solution
        self.solution = {'x': np.array(X_cand), 'y': np.array(Y_cand_pred)}
        return self.solution