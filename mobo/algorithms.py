from .mobo import MOBO

"""
High-level algorithm specifications by providing config
"""


class DGEMO(MOBO):
    """
    DGEMO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "identity",
        "solver": "discovery",
        "selection": "dgemo",
    }


class TSEMO(MOBO):
    """
    TSEMO
    """

    config = {
        "surrogate": "ts",
        "acquisition": "identity",
        "solver": "nsga2",
        "selection": "hvi",
    }


class USEMO_EI(MOBO):
    """
    USeMO, using EI as acquisition
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "nsga2",
        "selection": "uncertainty",
    }


class MOEAD_EGO(MOBO):
    """
    MOEA/D-EGO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "moead",
        "selection": "moead",
    }


class ParEGO(MOBO):
    """
    ParEGO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "parego",
        "selection": "random",
    }


"""
Define new algorithms here
"""


class Custom(MOBO):
    """
    Totally rely on user arguments to specify each component
    """

    config = None


class PSL(MOBO):
    config = {
        "surrogate": "gp",
        "acquisition": "identity",
        "solver": "psl",
        "selection": "hvi",
    }


class qNEHVI(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qnehvi",
        "selection": "identity",
    }


class qEHVI(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qehvi",
        "selection": "identity",
    }


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


NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
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
from botorch import fit_gpytorch_mll

from mobo.surrogate_model.base import SurrogateModel
from mobo.utils import safe_divide


from .surrogate_problem import SurrogateProblem
from .utils import Timer, find_pareto_front, calc_hypervolume
from .factory import init_from_config
from .transformation import StandardTransform
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples

rpr = torch.Tensor.__repr__

torch.manual_seed(0)

def _tensor_str(self):
    return "Tnsr{}".format(self.shape) + rpr(self)


torch.Tensor.__repr__ = _tensor_str

import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class qNEHVIStupid(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qnehvi",
        "selection": "hvi",
    }
    N_BATCH = 20 if not SMOKE_TEST else 10
    MC_SAMPLES = 128 if not SMOKE_TEST else 16

    verbose = True

    def __init__(self, problem, n_iter, ref_point, framework_args):
        """
        Input:
            problem: the original / real optimization problem
            n_iter: number of iterations to optimize
            ref_point: reference point for hypervolume calculation
            framework_args: arguments to initialize each component of the framework
        """
        super().__init__(problem, n_iter, ref_point, framework_args)

    def step(self):
        
        print(f"========== Iteration ==========")

        timer = Timer()
        self.transformation.fit(self.X, self.Y)
        train_x = torch.from_numpy(self.transformation.do(self.X)).to(**tkwargs)
        train_y = torch.from_numpy(self.Y).to(**tkwargs)
        
        mll, model = self.initialize_model(train_x, train_y)
        # fit the models
        fit_gpytorch_mll(mll)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qnehvi_sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([MC_SAMPLES])
        )

        new_x, new_aq = self.optimize_and_get_observation(model, train_x, train_y, qnehvi_sampler)
        
        self.solver.solution = {'x': np.array(new_x), 'y': np.array(new_aq)}
        
        #"select"
        new_X = torch.from_numpy(self.transformation.undo(new_x.numpy())).to(**tkwargs)
        
        
        new_Y = torch.from_numpy(self.real_problem.evaluate(new_X)).to(**tkwargs)

        X_next = new_X.numpy()
        Y_next = new_Y.numpy()
        
        # evaluate prediction of X_next on surrogate model
        # val = self.surrogate_model.evaluate(self.transformation.do(x=X_next), std=True)
        # Y_next_pred_mean = self.transformation.undo(y=val['F'])
        # Y_next_pred_std = val['S']
        # acquisition, _, _ = self.acquisition.evaluate(val)
        
        Y_next_pred_mean=model.posterior(new_x).mean.detach().cpu().numpy()
        Y_next_pred_std=model.posterior(new_x).variance.detach().cpu().numpy()
        acquisition = torch.stack([new_aq]*self.real_problem.n_obj).numpy()

        
        if self.real_problem.n_constr > 0:
            Y_next = Y_next[0]
        self._update_status(X_next, Y_next)
        timer.log("New samples evaluated")

        # statistics
        self.global_timer.log("Total runtime", reset=False)
        print(
            "Total evaluations: %d, hypervolume: %.4f\n"
            % (self.sample_num, self.status["hv"])
        )

        # return new data iteration by iteration
        return X_next, Y_next, Y_next_pred_mean, Y_next_pred_std, acquisition

    # def solveee(self, X_init_, Y_init_):
        
    #     """
    #     Solve the real multi-objective problem from initial data (X_init, Y_init)
    #     """
    #     # determine reference point from data if not specified by arguments
    #     self.ref_point = np.max(Y_init, axis=0)
    #     print("ref_point", self.ref_point)
        
    #     # self.ref_point = self.real_problem.ref_point
    #     self.solver.ref_point = np.min(Y_init, axis=0) # ref point is different for botorch
    #     print("ref_point solver", self.solver.ref_point)
        
    #     self._update_status(X_init, Y_init)

    #     global_timer = Timer()

    #     X_init = torch.from_numpy(X_init).to(**tkwargs)
    #     Y_init = torch.from_numpy(Y_init).to(**tkwargs)
    #     hvs = []

    #     train_X, train_Y, = X_init, Y_init
        


    #     # run N_BATCH rounds of BayesOpt after the initial random batch
    #     for iteration in range(self.N_BATCH):
    #         t0 = time.monotonic()

    #         self.transformation.fit(self.X, self.Y)
    #         train_x = torch.from_numpy(self.transformation.do(self.X)).to(**tkwargs)
    #         train_y = torch.from_numpy(self.Y).to(**tkwargs)
            
    #         mll, model = self.initialize_model(train_x, train_y)
    #         # fit the models
    #         fit_gpytorch_mll(mll)

    #         # define the qEI and qNEI acquisition modules using a QMC sampler
    #         qnehvi_sampler = SobolQMCNormalSampler(
    #             sample_shape=torch.Size([MC_SAMPLES])
    #         )

    #         new_x, new_aq = self.optimize_and_get_observation(model, train_x, train_y, qnehvi_sampler)
            
    #         self.solver.solution = {'x': np.array(new_x), 'y': np.array(new_aq)}
            
    #         #"select"
    #         new_X = torch.from_numpy(self.transformation.undo(new_x.numpy())).to(**tkwargs)
            
            
    #         new_Y = torch.from_numpy(self.real_problem.evaluate(new_X)).to(**tkwargs)

    #         X_next = new_X.numpy()
    #         Y_next = new_Y.numpy()
    #         self._update_status(X_next, Y_next)
            
    #         train_X = torch.cat([train_X, new_X])
    #         train_Y = torch.cat([train_Y, new_Y])
    #         # update progress
    #         for hvs_list_i, train_Y_i in zip((hvs,), (train_Y,)):
    #             # compute hypervolume
    #             bd = DominatedPartitioning(
    #                 ref_point=torch.from_numpy(self.solver.ref_point).to(**tkwargs),
    #                 Y=train_Y_i
    #             )
    #             volume = bd.compute_hypervolume().item()
    #             hvs_list_i.append(volume)

    #         # reinitialize the models so they are ready for fitting on next iteration
    #         # Note: we find improved performance from not warm starting the model hyperparameters
    #         # using the hyperparameters from the previous iteration

    #         t1 = time.monotonic()

    #         if verbose:
    #             print(
    #                 f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = "
    #                 f"( {hvs[-1]:>4.2f}), "
    #                 f"time = {t1-t0:>4.2f}.",
    #                 end="",
    #             )
    #         else:
    #             print(".", end="")

    #         yield new_X.numpy(), new_Y.numpy()


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

    def optimize_and_get_observation(self, model, train_x, train_y, sampler):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        # partition non-dominated space into disjoint rectangles
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.solver.ref_point,  # use known reference point
            X_baseline=torch.from_numpy(self.transformation.do(train_x.numpy())).to(**tkwargs),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )        

        standard_bounds = torch.zeros(2, self.real_problem.n_var, **tkwargs)
        standard_bounds[1] = 1
        
        new_x, new_aq = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.solver.batch_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        return new_x, new_aq

def get_algorithm(name):
    """
    Get class of algorithm by name
    """
    algo = {
        "dgemo": DGEMO,
        "tsemo": TSEMO,
        "usemo-ei": USEMO_EI,
        "moead-ego": MOEAD_EGO,
        "parego": ParEGO,
        "custom": Custom,
        "psl": PSL,
        "qnehvi": qNEHVI,
        "qehvi": qEHVI,
        "qnehvistupid": qNEHVIStupid,
    }
    return algo[name]
