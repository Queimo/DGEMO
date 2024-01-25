import numpy as np
from .problem import RiskyProblem, Problem

class K1(RiskyProblem):

    def __init__(self, sigma=0., repeat_eval=1):
        
        self.sigma = sigma
        self.repeat_eval = repeat_eval
        self.bounds = np.array([[0.5, 0.0], [3.5, 1.0]])
        self.dim = 2
        self.num_objectives = 2
        self.max_hv = 16
        
        super().__init__(
            n_var=self.dim, 
            n_obj=self.num_objectives, 
            n_constr=0,
            xl=self.bounds[0,:],
            xu=self.bounds[1,:],
        )
    
    def _evaluate_F(self, x):
        train_obj = self.evaluate_repeat(x)
        # train_obj = np.quantile(train_obj, 0.01, axis=-1)
        train_obj = train_obj.mean(axis=-1)
        return -1 * train_obj / np.array([18., 1.]) + np.array([0., 1.])
    
    def _evaluate_rho(self, x):
        train_rho = self.evaluate_repeat(x).std(axis=-1)
        #check nan
        if np.isnan(train_rho).any():
            print("nan in rho")
            train_rho = np.zeros_like(train_rho)
        return train_rho / np.array([18., 1.]) + 1.1212432443345e-04 #introduce numerical love
    
    
    def pareto_front(self, n_pareto_points=500):
        
        from .common import generate_initial_samples, get_problem
        from mobo.solver import NSGA2Solver
        from arguments import get_solver_args
         
        prob = self.__class__(repeat_eval=100)
        X_init, Y_init, rho_init = generate_initial_samples(prob, n_pareto_points)
        
        Y_l = self.evaluate_repeat(X_init).min(axis=-1)
        Y_h = self.evaluate_repeat(X_init).max(axis=-1)
        
        #namespace to dict
        solver_args = vars(get_solver_args())
        
        
        solver = NSGA2Solver(**solver_args)

        # find Pareto front
        solution = solver.solve(prob, X_init, Y_init)
        solution_l = solver.solve(prob, X_init, Y_l)
        solution_h = solver.solve(prob, X_init, Y_h)
        
        Y_paretos = solution['y']
        Y_paretos_l = solution_l['y']
        Y_paretos_h = solution_h['y']
        return [Y_paretos, Y_paretos_l, Y_paretos_h]
    
    def evaluate_repeat(self, x: np.array) -> np.array:
        y_true = self.f(x)
        sigmas = self.get_noise_var(x)
        y_true = np.stack([y_true] * self.repeat_eval, axis=-1)
        y = y_true - np.expand_dims(sigmas, -1) * np.power(np.random.randn(*y_true.shape), 2)
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
            (7.0217 / (1 + np.exp(-13.2429 * (x[:, 0] - 2.1502))) + 6.2086 - 9) ** 2
        )

        troughput = (
            (x[:, 1] / self.bounds[1][1]) ** 0.5
            * 1
            / (1 + np.exp(-13.2429 * (x[:, 0] - 2.1502)))
        ) \
            # * 0 + 1. #DANGER

        return np.stack([pH, troughput]).T 

    def get_noise_var(self, x):
        # bell-shaped noise centered at 1.5
        # \frac{1}{1+e^{20\left(x-1\right)}}
        inner = 1/(1 + np.exp(20 * (x[:, 1] - 0.9)))
        
        sigmas_pH = self.sigma * np.exp(-20 * (x[:, 0] - 2.15) ** 2)
        sigmas_pH *= np.power(x[:, 1], 4) * 4 * inner

        sigmas_troughput = self.sigma / 200 * np.ones_like(x[:, 1])

        return np.stack([sigmas_pH, sigmas_troughput]).T


class K2(K1):
    def __init__(self, sigma=0.2, repeat_eval=3):
        super().__init__(
            sigma=sigma,
            repeat_eval=repeat_eval
        )

class K3(K1):
    def __init__(self, sigma=0.5, repeat_eval=3):
        super().__init__(
            sigma=sigma,
            repeat_eval=repeat_eval
        )

class K4(K1):
    def __init__(self, sigma=0.7, repeat_eval=3):
        super().__init__(
            sigma=sigma,
            repeat_eval=repeat_eval
        )

class K5(K1):
    def __init__(self, sigma=1.0, repeat_eval=3):
        super().__init__(
            sigma=sigma,
            repeat_eval=repeat_eval
        )

class K6(K1):
    def __init__(self, sigma=0.2, repeat_eval=1):
        super().__init__(
            sigma=sigma,
            repeat_eval=repeat_eval
        )

class K7(K1):
    def __init__(self, sigma=0.5, repeat_eval=1):
        super().__init__(
            sigma=sigma,
            repeat_eval=repeat_eval
        )

class K8(K1):
    def __init__(self, sigma=0.7, repeat_eval=1):
        super().__init__(
            sigma=sigma,
            repeat_eval=repeat_eval
        )

class K9(K1):
    def __init__(self, sigma=1.0, repeat_eval=1):
        super().__init__(
            sigma=sigma,
            repeat_eval=repeat_eval
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    prob = K1()
    true_front = prob.pareto_front()
    print(true_front)
    
    #plot pareto front
    plt.scatter(true_front[:,0], true_front[:,1])
    plt.show()
    