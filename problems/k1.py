import numpy as np
from .problem import RiskyProblem
import torch
from torch import Tensor

class K1(RiskyProblem):

    def __init__(self):
        
        self.sigma = 0.
        self.repeat_eval = 1
        self.bounds = torch.tensor([[0.5, 0.0], [3.5, 1.0]])
        self.dim = 2
        self.num_objectives = 2
        self.max_hv = 16
        
        super().__init__(
            n_var=self.dim, 
            n_obj=self.num_objectives, 
            n_constr=0,
            xl=self.bounds[0,:].numpy(),
            xu=self.bounds[1,:].numpy(),
        )
    
    def _evaluate_F(self, x):
        x_torch = torch.from_numpy(x).float()
        train_obj = self.evaluate_repeat(x_torch).mean(dim=-1)
        return -1 * train_obj.numpy() / np.array([18., 1.]) + np.array([0., 1.])
    
    def _evaluate_rho(self, x):
        x_torch = torch.from_numpy(x).float()
        train_rho = self.evaluate_repeat(x_torch).std(dim=-1)
        #check nan
        if torch.isnan(train_rho).any():
            print("nan in rho")
            train_rho = torch.zeros_like(train_rho)
        return train_rho.numpy() / np.array([18., 1.]) + 1.1212432443345e-04 #introduce numerical love
    
    
    def _calc_pareto_front(self, n_pareto_points=500):
        
        from .common import generate_initial_samples, get_problem
        from mobo.solver import NSGA2Solver
        from arguments import get_solver_args
         
        prob = get_problem('k1')
        X_init, Y_init, rho_init = generate_initial_samples(prob, n_pareto_points)
        
        #namespace to dict
        solver_args = vars(get_solver_args())
        
        
        solver = NSGA2Solver(**solver_args)

        # find Pareto front
        solution = solver.solve(prob, X_init, Y_init)
        
        Y_paretos = solution['y']
        return Y_paretos
    
    def evaluate_repeat(self, x: Tensor) -> Tensor:
        y_true = self.f(x)
        sigmas = self.get_noise_var(x)
        y_true = torch.stack([y_true] * self.repeat_eval, dim=-1)
        y = y_true - sigmas.unsqueeze(-1) * torch.pow(torch.randn_like(y_true), 2)
        return y

    
    # def evaluate_on_test(self, x: Tensor) -> Tensor:
    #     y_true = self.f(x)
    #     sigmas = self.get_noise_var(x)

    #     shape = y_true.shape
    #     noise = sigmas * torch.randn(
    #         shape, generator=torch.Generator().manual_seed(self.seed_test)
    #     )
    #     y = y_true + noise
    #     return y

    def get_domain(self):
        return self.bounds


    def f(self, x):
        pH = -(
            (7.0217 / (1 + torch.exp(-13.2429 * (x[:, 0] - 2.1502))) + 6.2086 - 9) ** 2
        )

        troughput = (
            (x[:, 1] / self.bounds[1][1]) ** 0.5
            * 1
            / (1 + torch.exp(-13.2429 * (x[:, 0] - 2.1502)))
        ) \
            # * 0 + 1. #DANGER

        return torch.stack([pH, troughput]).T 

    def get_noise_var(self, x):
        # bell-shaped noise centered at 1.5
        sigmas_pH = self.sigma * torch.exp(-20 * (x[:, 0] - 2.15) ** 2)
        sigmas_pH *= x[:, 1] * x[:, 1] * 4

        sigmas_troughput = self.sigma / 200 * torch.ones_like(x[:, 1])

        return torch.stack([sigmas_pH, sigmas_troughput]).T

    def get_info_to_dump(self, x):
        dict_to_dump = {
            "f": self.f(x).squeeze(),
            "rho": self.get_noise_var(x).squeeze(),
        }

        return dict_to_dump

class K2(K1):
    def __init__(self):
        super().__init__()
        self.sigma = 0.2
        self.repeat_eval = 3
        
class K3(K1):
    def __init__(self):
        super().__init__()
        self.sigma = 0.5
        self.repeat_eval = 3

class K4(K1):
    def __init__(self):
        super().__init__()
        self.sigma = 0.7
        self.repeat_eval = 3
        
class K5(K1):
    def __init__(self):
        super().__init__()
        self.sigma = 1.
        self.repeat_eval = 3

class K6(K1):
    def __init__(self):
        super().__init__()
        self.sigma = 0.2
        self.repeat_eval = 1
        
class K7(K1):
    def __init__(self):
        super().__init__()
        self.sigma = 0.5
        self.repeat_eval = 1
        
class K8(K1):
    def __init__(self):
        super().__init__()
        self.sigma = 0.7
        self.repeat_eval = 1
        
class K9(K1):
    def __init__(self):
        super().__init__()
        self.sigma = 1.
        self.repeat_eval = 1

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    prob = K1()
    true_front = prob.pareto_front()
    print(true_front)
    
    #plot pareto front
    plt.scatter(true_front[:,0], true_front[:,1])
    plt.show()
    