import numpy as np
from .problem import RiskyProblem
import pandas as pd


class Experiment(RiskyProblem):

    def __init__(self, sigma=0.5, repeat_eval=3):

        self.sigma = np.nan
        self.bounds = np.array([[0.5, 0.0], [3.5, 1.0]])
        self.dim = 2
        self.num_objectives = 3
        df = pd.read_excel("./problems/data/XRD+synthsis_data_b3.xlsx")
        df = df[["id", "C_ZnCl", "C_NaOH/C_ZnCl", "Aspect Ratio", "Peak Ratio"]]
        df_mean = df.select_dtypes(include=["float64", "int64"]).groupby("id").mean()
        df_std = df.select_dtypes(include=["float64", "int64"]).groupby("id").std()
        df_mean_std = pd.merge(
            df_mean,
            df_std,
            left_index=True,
            right_index=True,
            suffixes=("_mean", "_std"),
        )
        self.df_mean_std = df_mean_std
        print(df_mean_std)
        # X1 = C_NaOH/C_ZnCl, X2 = C_ZnCl
        # Y1 = Peak Ratio, Y2 = Aspect Ratio, Y3 = C_ZnCl
        self.X = df_mean_std[["C_NaOH/C_ZnCl_mean", "C_ZnCl_mean"]].values
        self.Y = (
            -1
            * df_mean_std[
                ["Peak Ratio_mean", "Aspect Ratio_mean", "C_ZnCl_mean"]
            ].values
        )  # we assume minimzation
        self.rho = df_mean_std[
            ["Peak Ratio_std", "Aspect Ratio_std", "C_ZnCl_std"]
        ].values

        super().__init__(
            n_var=self.dim,
            n_obj=self.num_objectives,
            n_constr=0,
            xl=self.bounds[0, :],
            xu=self.bounds[1, :],
        )

    def _evaluate_F(self, x):
        return self.Y[: x.shape[0], :]

    def _evaluate_rho(self, x):
        return self.rho

    def pareto_front(self, n_pareto_points=1000):

        from mobo.utils import find_pareto_front

        Y_paretos = find_pareto_front(self.Y)
        Y_paretos_l = find_pareto_front(self.Y)
        Y_paretos_h = find_pareto_front(self.Y)

        return [Y_paretos, Y_paretos_l, Y_paretos_h]

    def get_domain(self):
        return self.bounds

    def f(self, X):
        return self.Y

    def get_noise_var(self, X):
        return self.rho


class Experiment4D(RiskyProblem):

    def __init__(self, sigma=0.5, repeat_eval=3):

        self.sigma = np.nan
        #                       C_NaOH/C_ZnCl, C_ZnCl, Q_AC, Q_Air
        self.bounds = np.array([[0.5, 0.0, 8.0, 1.0], 
                                [3.5, 1.0, 20.0, 3.0]])
        self.dim = 4
        self.num_objectives = 3
        df = pd.read_excel("./problems/data/XRD+synthsis_data_b3.xlsx")
        df = df[["id", "C_ZnCl", "C_NaOH/C_ZnCl", "C_NaOH" ,"Aspect Ratio", "Peak Ratio", "Q_AC", "Q_Air", "N_ZnO"]]
        
        df_mean = df.select_dtypes(include=["float64", "int64"]).groupby("id").mean()
        df_std = df.select_dtypes(include=["float64", "int64"]).groupby("id").std()
        df_mean_std = pd.merge(
            df_mean,
            df_std,
            left_index=True,
            right_index=True,
            suffixes=("_mean", "_std"),
        )
        self.df_mean_std = df_mean_std
        print(df_mean_std)
        # X1 = C_NaOH/C_ZnCl, X2 = C_ZnCl
        # Y1 = Peak Ratio, Y2 = Aspect Ratio, Y3 = C_ZnCl
        self.X = df_mean_std[["C_NaOH/C_ZnCl_mean", "C_ZnCl_mean"]].values
        self.Y = (
            -1
            * df_mean_std[
                ["Peak Ratio_mean", "Aspect Ratio_mean", "N_ZnO_mean"]
            ].values
        )  # we assume minimzation
        self.rho = df_mean_std[
            ["Peak Ratio_std", "Aspect Ratio_std", "N_ZnO_std"]
        ].values

        super().__init__(
            n_var=self.dim,
            n_obj=self.num_objectives,
            n_constr=0,
            xl=self.bounds[0, :],
            xu=self.bounds[1, :],
        )

    def _evaluate_F(self, x):
        return self.Y[: x.shape[0], :]

    def _evaluate_rho(self, x):
        return self.rho

    def pareto_front(self, n_pareto_points=1000):

        from mobo.utils import find_pareto_front

        Y_paretos = find_pareto_front(self.Y)
        Y_paretos_l = find_pareto_front(self.Y)
        Y_paretos_h = find_pareto_front(self.Y)

        return [Y_paretos, Y_paretos_l, Y_paretos_h]

    def get_domain(self):
        return self.bounds

    def f(self, X):
        return self.Y

    def get_noise_var(self, X):
        return self.rho


if __name__ == "__main__":

    prob = Exp()
