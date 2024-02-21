import numpy as np

# from ..problems import RiskyProblem
from mobo.utils import calculate_var

class Peaks:

    def __init__(self, sigma=0.5, repeat_eval=1):

        self.sigma = sigma
        self.repeat_eval = repeat_eval
        self.bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.dim = 2
        self.num_objectives = 2

        # super().__init__(
        #     n_var=self.dim,
        #     n_obj=self.num_objectives,
        #     n_constr=0,
        #     xl=self.bounds[0,:],
        #     xu=self.bounds[1,:],
        # )

    def _evaluate_F(self, x):
        train_obj = self.evaluate_repeat(x)
        # train_obj = np.quantile(train_obj, 0.01, axis=-1)
        train_obj = train_obj.mean(axis=-1)
        return train_obj

    def _evaluate_rho(self, x):
        train_rho = self.evaluate_repeat(x).std(axis=-1)
        # check nan
        if np.isnan(train_rho).any():
            # print("nan in rho")
            train_rho = np.zeros_like(train_rho)
        return train_rho

    def pareto_front(self, n_pareto_points=500):

        from .common import generate_initial_samples, get_problem
        from mobo.solver import NSGA2Solver
        from arguments import get_solver_args

        prob = self.__class__(repeat_eval=100)
        X_init, Y_init, rho_init = generate_initial_samples(prob, n_pareto_points)

        Y_l = self.evaluate_repeat(X_init).min(axis=-1)
        Y_h = self.evaluate_repeat(X_init).max(axis=-1)

        # namespace to dict
        solver_args = vars(get_solver_args())

        solver = NSGA2Solver(**solver_args)

        # find Pareto front
        solution = solver.solve(prob, X_init, Y_init)
        solution_l = solver.solve(prob, X_init, Y_l)
        solution_h = solver.solve(prob, X_init, Y_h)

        Y_paretos = solution["y"]
        Y_paretos_l = solution_l["y"]
        Y_paretos_h = solution_h["y"]
        return [Y_paretos, Y_paretos_l, Y_paretos_h]

    # def pareto_front(self, n_pareto_points=1000):

    #     from .common import generate_initial_samples
    #     from mobo.utils import find_pareto_front

    #     prob = self.__class__(repeat_eval=50)
    #     X_init, Y_init, rho_init = generate_initial_samples(prob, n_pareto_points)

    #     Y_l = np.quantile(self.evaluate_repeat(X_init), 0.9, axis=-1)
    #     Y_h = np.quantile(self.evaluate_repeat(X_init), 0.1, axis=-1)
    #     # Y_h = self.evaluate_repeat(X_init).max(axis=-1)

    #     Y_paretos = find_pareto_front(Y_init)
    #     Y_paretos_l = find_pareto_front(Y_l)
    #     Y_paretos_h = find_pareto_front(Y_h)

    #     return [Y_paretos, Y_paretos_l, Y_paretos_h]

    def evaluate_repeat(self, x: np.array) -> np.array:
        y_true = self.f(x)
        sigmas = self.get_noise_var(x)
        y_true = np.stack([y_true] * self.repeat_eval, axis=-1)
        y = y_true + np.expand_dims(sigmas, -1) * np.random.randn(*y_true.shape)
        return y

    def get_domain(self):
        return self.bounds

    def styblinski_tang_function(self, x1, x2):
        x1 = 10 * x1 - 5
        x2 = 10 * x2 - 5
        return (
            0.5 * ((x1**4 - 16 * x1**2 + 5 * x1) + (x2**4 - 16 * x2**2 + 5 * x2)) / 250
        )

    def brannin_function(self, x1, x2):
        x1 = 15 * x1 - 5
        x2 = 15 * x2
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return (
            a
            * ((x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)
            / 300
        )

    def f(self, X):
        x1, x2 = X[:, 0], X[:, 1]
        y1 = self.brannin_function(x1, x2)
        y2 = self.styblinski_tang_function(x1, x2)

        return np.stack([y1, y2], axis=-1)

    def sigmoid(self, x):
        """Sigmoid function for scaling."""
        return 1 / (1 + np.exp(-x))

    def get_noise_var(self, X):

        x1, x2 = X[:, 0], X[:, 1]
        # noise_factor = self.sigmoid(20 * (x1 - 0.6)) * self.sigmoid(20 * (-x2 + 0.4))
        noise_factor = self.sigmoid(20 * (-x2 + 0.4))
        rho1 = self.sigma * noise_factor
        rho2 = self.sigma * noise_factor

        return np.stack([rho1, rho2], axis=-1)


# class K2(K1):
#     def __init__(self, sigma=0.2, repeat_eval=3):
#         super().__init__(
#             sigma=sigma,
#             repeat_eval=repeat_eval
#         )


class PeaksS5R3(Peaks):
    def __init__(self):
        super().__init__(sigma=0.5, repeat_eval=3)


class Peaks4D:

    def __init__(self, sigma=0.0, repeat_eval=1):

        self.sigma = sigma
        self.repeat_eval = repeat_eval
        self.bounds = np.array([[0.0] * 4, [1.0] * 4])  # Adjusted for 6 dimensions
        self.dim = 4  # Adjusted for 6 dimensions
        self.num_objectives = 2

        super().__init__(
            n_var=self.dim,
            n_obj=self.num_objectives,
            n_constr=0,
            xl=self.bounds[0, :],
            xu=self.bounds[1, :],
        )

    # The _evaluate_F, _evaluate_rho, pareto_front, evaluate_repeat, and get_domain methods remain largely unchanged.
    # Minor adjustments might be needed for handling 6-dimensional inputs in evaluate_repeat if the f function changes its input handling.

    def _evaluate_F(self, x):
        train_obj = self.evaluate_repeat(x)
        # train_obj = np.quantile(train_obj, 0.01, axis=-1)
        train_obj = train_obj.mean(axis=-1)
        return train_obj

    def _evaluate_rho(self, x):
        train_rho = self.evaluate_repeat(x).var(axis=-1)
        # check nan
        if np.isnan(train_rho).any():
            # print("nan in rho")
            train_rho = np.zeros_like(train_rho)
        return train_rho

    def pareto_front(self, n_pareto_points=500):

        from .common import generate_initial_samples, get_problem
        from mobo.solver import NSGA2Solver
        from arguments import get_solver_args

        prob = self.__class__(repeat_eval=100)
        X_init, Y_init, rho_init = generate_initial_samples(prob, n_pareto_points)

        Y_l = self.evaluate_repeat(X_init).min(axis=-1)
        Y_h = self.evaluate_repeat(X_init).max(axis=-1)

        # namespace to dict
        solver_args = vars(get_solver_args())

        solver = NSGA2Solver(**solver_args)

        # find Pareto front
        solution = solver.solve(prob, X_init, Y_init)
        solution_l = solver.solve(prob, X_init, Y_l)
        solution_h = solver.solve(prob, X_init, Y_h)

        Y_paretos = solution["y"]
        Y_paretos_l = solution_l["y"]
        Y_paretos_h = solution_h["y"]
        return [Y_paretos, Y_paretos_l, Y_paretos_h]

    def evaluate_repeat(self, x: np.array) -> np.array:
        y_true = self.f(x)
        sigmas = self.get_noise_var(x)
        y_true = np.stack([y_true] * self.repeat_eval, axis=-1)
        y = y_true + np.expand_dims(sigmas, -1) * np.random.randn(*y_true.shape)
        return y

    def f(self, X):
        # Adjusted for 6-dimensional inputs
        y1 = self.brannin_function(X[:, :2])  # Use the first 2 dimensions for Brannin
        y2 = self.styblinski_tang_function(
            X[:, 2:]
        )  # Use the remaining dimensions for Styblinski-Tang
        return np.stack([y1, y2], axis=-1)

    def styblinski_tang_function(self, X):
        # Adjusted for handling multiple dimensions beyond the original two
        X_scaled = 10 * X - 5
        return 0.5 * np.sum(X_scaled**4 - 16 * X_scaled**2 + 5 * X_scaled, axis=1) / 250

    def brannin_function(self, X):
        # Only uses the first two dimensions from the input
        x1, x2 = X[:, 0], X[:, 1]
        x1 = 15 * x1 - 5
        x2 = 15 * x2
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        return (
            a
            * ((x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)
            / 300
        )

    def sigmoid(self, x):
        """Sigmoid function for scaling."""
        return 1 / (1 + np.exp(-x))

    def get_noise_var(self, X):
        # Adjust noise variance calculation for 6 dimensions, if necessary
        # Example using a simplified noise model based on first two dimensions
        x1, x2 = X[:, 0], X[:, 1]
        noise_factor = self.sigmoid(20 * (-x2 + 0.4))
        rho = self.sigma * noise_factor

        # Apply the same noise factor to all outputs for simplicity
        # This part can be customized based on the desired noise model for the 6D problem
        return np.stack(
            [rho, rho], axis=-1
        )  # Keep this aligned with the number of objectives


if __name__ == "__main__":

    prob = Peaks(sigma=.5, repeat_eval=3)

    n = 20

    # create a surface plot with plotly of the objective functions
    import plotly.graph_objects as go

    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    x1, x2 = np.meshgrid(x1, x2)
    X = np.stack([x1.flatten(), x2.flatten()]).T
    Y = prob.f(X).reshape(n, n, 2)
    rho_measured = prob._evaluate_rho(X).reshape(n, n, 2)
    rho_real = prob.get_noise_var(X).reshape(n, n, 2)
    Y_noise = prob._evaluate_F(X).reshape(n, n, 2)

    var = calculate_var(Y.flatten(), variance=rho_real.flatten())
    var = var.reshape(n, n, 2)

    fig = go.Figure(
        data=[
            go.Surface(x=x1, y=x2, z=Y[:, :, 0]),
            go.Surface(x=x1, y=x2, z=Y[:, :, 1]),
        ]
    )
    # fig.show()

    # create a subplot with 2 countour plots

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    r = 2
    c = 3
    # Titles in LaTeX

    # ("$F_1$", "$F_2$", "$\\rho_{1}$", "$\\rho_{1}$", "$VaR(F_1)$", "$VaR(F_2)$")
    # change order of plot titles

    fig = make_subplots(
        rows=r,
        cols=c,
        subplot_titles=(
            "$F_1$",
            "$\\rho_{1}$",
            "$VaR_{0.9}(F_1)$",
            "$F_2$",
            "$\\rho_{2}$",
            "$VaR_{0.9}(F_2)$",
        ),
        shared_xaxes=True,
        shared_yaxes=True,
    )
    
    #increase font size of latex titles
    for i in range(1, r+1):
        for j in range(1, c+1):
            fig.update_xaxes(title_font_size=30, row=i, col=j)
            fig.update_yaxes(title_font_size=30, row=i, col=j)
   
    # Y_list = [Y[:,:,0], Y[:,:,1], Y_noise[:,:,0], Y_noise[:,:,1], rho_real[:,:,0], rho_real[:,:,1], rho_measured[:,:,0], rho_measured[:,:,1]]

    Y_list = [
        Y[:, :, 0],
        Y[:, :, 1],
        rho_real[:, :, 0],
        rho_real[:, :, 1],
        var[:, :, 0],
        var[:, :, 1],
    ]

    # titles = ["y1", "y2", "y1 + noise", "y2 + noise", "rho_real1", "rho_real2", "rho_measured1", "rho_measured2"]
    titles = ["y1", "y2", "rho_measured1", "rho_measured2", "VaR1", "VaR2"]

    for i, y in enumerate(Y_list):
        row = (i % r) + 1  # Corrected row index
        col = (i // r) + 1  # Corrected column index

        fig.add_trace(
            go.Contour(
                x=x1[0, :],
                y=x2[:, 0],
                z=y,
                colorscale="Viridis",
                showscale=False,
                ncontours=20,
            ),
            row=row,
            col=col,
        )
    fig.update_traces(ncontours=45, selector=dict(type='contour'))

    # fig.update_layout(
    #     margin=dict(
    #         l=50,
    #         r=50,
    #         b=100,
    #         t=100,
    #         pad=4
    #     ),
    # )

    fig.show()
