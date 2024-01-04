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
        "selection": "hvi",
    }


class qEHVI(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qehvi",
        "selection": "hvi",
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


BATCH_SIZE = 1
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


def _tensor_str(self):
    return "Tnsr{}".format(self.shape) + rpr(self)


torch.Tensor.__repr__ = _tensor_str


class qNEHVIStupid(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qnehvi",
        "selection": "hvi",
    }
    config = {}

    def __init__(self, problem, n_iter, ref_point, framework_args):
        """
        Input:
            problem: the original / real optimization problem
            n_iter: number of iterations to optimize
            ref_point: reference point for hypervolume calculation
            framework_args: arguments to initialize each component of the framework
        """
        super().__init__(problem, n_iter, ref_point, framework_args)

    def _update_status(self, X, Y):
        """
        Update the status of algorithm from data
        """
        if self.sample_num == 0:
            self.X = X
            self.Y = Y
        else:
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])
        self.sample_num += len(X)

        self.status["pfront"], pfront_idx = find_pareto_front(self.Y, return_index=True)
        self.status["pset"] = self.X[pfront_idx]
        self.status["hv"] = calc_hypervolume(self.status["pfront"], self.ref_point)

    def solve(self, X_init, Y_init):
        X_init = torch.from_numpy(X_init).to(**tkwargs)
        Y_init = torch.from_numpy(Y_init).to(**tkwargs).unsqueeze(-1)

        standard_bounds = torch.zeros(2, self.real_problem.dim, **tkwargs)
        standard_bounds[1] = 1

        def generate_initial_data(n=5):
            # generate training data
            # train_x = draw_sobol_samples(self.real_problem.bounds, n, q=1).squeeze().to(**tkwargs)
            # return train_x, *K1_wrapper(train_x)

            return X_init, Y_init.mean(dim=-1), Y_init.mean(dim=-1)

        def K1_wrapper(x):
            train_obj_true = self.real_problem.f(x).to(**tkwargs)
            train_obj = self.real_problem.evaluate_repeat(x).to(**tkwargs)
            return train_obj.mean(dim=-1), train_obj_true

        def initialize_model(train_x, train_obj):
            # define models for objective and constraint
            train_x = normalize(train_x, self.real_problem.bounds)
            train_obj_mean = train_obj
            # train_obj_var = self.real_problem.evaluate(train_x).to(**tkwargs).var(dim=-1)
            train_obj_var = torch.zeros_like(train_obj).to(**tkwargs)
            models = []
            for i in range(train_obj.shape[1]):
                train_y = train_obj_mean[..., i]
                train_yvar = train_obj_var[..., i]
                models.append(
                    FixedNoiseGP(
                        train_x,
                        train_y.unsqueeze(-1),
                        train_yvar.unsqueeze(-1),
                        outcome_transform=Standardize(m=1),
                    )
                )
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            return mll, model

        def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
            """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
            # partition non-dominated space into disjoint rectangles
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.real_problem.ref_point.tolist(),  # use known reference point
                X_baseline=normalize(train_x, self.real_problem.bounds),
                prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
                sampler=sampler,
            )
            # optimize
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )
            # observe new values
            new_x = unnormalize(candidates.detach(), bounds=self.real_problem.bounds)
            new_obj, new_obj_true = K1_wrapper(new_x)
            return new_x, new_obj, new_obj_true

        from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

        # %%

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

        N_BATCH = 20 if not SMOKE_TEST else 10
        MC_SAMPLES = 128 if not SMOKE_TEST else 16

        verbose = True

        hvs_qnehvi = []

        # call helper functions to generate initial training data and initialize model
        (
            train_x_qparego,
            train_obj_qparego,
            train_obj_true_qparego,
        ) = generate_initial_data(n=2 * (self.real_problem.dim + 1))
        train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = (
            train_x_qparego,
            train_obj_qparego,
            train_obj_true_qparego,
        )
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

        # compute hypervolume
        bd = DominatedPartitioning(
            ref_point=self.real_problem.ref_point, Y=train_obj_true_qparego
        )
        volume = bd.compute_hypervolume().item()

        hvs_qnehvi.append(volume)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):
            t0 = time.monotonic()

            # fit the models
            fit_gpytorch_mll(mll_qnehvi)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qnehvi_sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([MC_SAMPLES])
            )

            (
                new_x_qnehvi,
                new_obj_qnehvi,
                new_obj_true_qnehvi,
            ) = optimize_qnehvi_and_get_observation(
                model_qnehvi, train_x_qnehvi, train_obj_qnehvi, qnehvi_sampler
            )

            train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
            train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
            train_obj_true_qnehvi = torch.cat(
                [train_obj_true_qnehvi, new_obj_true_qnehvi]
            )
            # update progress
            for hvs_list, train_obj in zip((hvs_qnehvi,), (train_obj_true_qnehvi,)):
                # compute hypervolume
                bd = DominatedPartitioning(
                    ref_point=self.real_problem.ref_point, Y=train_obj
                )
                volume = bd.compute_hypervolume().item()
                hvs_list.append(volume)

            # reinitialize the models so they are ready for fitting on next iteration
            # Note: we find improved performance from not warm starting the model hyperparameters
            # using the hyperparameters from the previous iteration
            mll_qnehvi, model_qnehvi = initialize_model(
                train_x_qnehvi, train_obj_qnehvi
            )

            t1 = time.monotonic()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = "
                    f"( {hvs_qnehvi[-1]:>4.2f}), "
                    f"time = {t1-t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")


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
