from . import NSGA2Solver, Solver
from pymoo.algorithms.nsga2 import NSGA2


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


class qNEHVISolver(NSGA2Solver):
    '''
    Solver based on PSL
    '''
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        

    def solve(self, problem, X, Y, rho):
        standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
        standard_bounds[1] = 1
        surrogate_model = problem.surrogate_model
        
        ref_point = self.ref_point
        print("ref_point", ref_point)
        
        # use nsga2 to find pareto front for later plots, has limitations
        self.solution = super().solve(problem, X, Y) 
        
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
        
        X_cand, Y_cand_pred = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.batch_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        
        selection = {'x': np.array(X_cand), 'y': np.array(Y_cand_pred)}
        
        return selection


class qEHVISolver(NSGA2Solver):
    '''
    Solver based on PSL
    '''
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        

    def solve(self, problem, X, Y, rho):
        standard_bounds = torch.zeros(2, problem.n_var, **tkwargs)
        standard_bounds[1] = 1
        surrogate_model = problem.surrogate_model
        
        ref_point = self.ref_point
        print("ref_point", ref_point)
        
        # use nsga2 to find pareto front for later plots, has limitations
        self.solution = super().solve(problem, X, Y)
        
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
        acq_func = qExpectedHypervolumeImprovement(
            ref_point=torch.tensor(ref_point).to(**tkwargs),
            model=surrogate_model.bo_model,
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
        selection = {'x': np.array(X_cand), 'y': np.array(Y_cand_pred)}
        
        return selection