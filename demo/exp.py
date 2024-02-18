import numpy as np
# from .problem import RiskyProblem
import pandas as pd

class Exp:

    def __init__(self, sigma=.5, repeat_eval=3):
        
        self.sigma = np.nan
        self.repeat_eval = 3
        self.bounds = np.array([[0.5, 0.0], [3.5, 1.0]])
        self.dim = 2
        self.num_objectives = 3
        df = pd.read_excel("./problems/data/XRD+synthsis_data.xlsx")
        df = df[["id", "C_ZnCl", "C_NaOH/C_ZnCl", "Aspect Ratio", "Peak Ratio"]]
        df_mean = df.select_dtypes(include=['float64', 'int64']).groupby("id").mean()
        df_std = df.select_dtypes(include=['float64', 'int64']).groupby("id").std()
        df_mean_std = pd.merge(df_mean, df_std, left_index=True, right_index=True, suffixes=("_mean", "_std"))
        self.df_mean_std = df_mean_std
        print(df_mean_std)
        self.X = df_mean_std[["C_NaOH/C_ZnCl_mean", "C_ZnCl_mean"]].values
        self.Y = df_mean_std[["Peak Ratio_mean", "Aspect Ratio_mean", "C_ZnCl_mean"]].values
        self.rho = df_mean_std[["Peak Ratio_std", "Aspect Ratio_std", "C_ZnCl_std"]].values
        
        # super().__init__(
        #     n_var=self.dim, 
        #     n_obj=self.num_objectives, 
        #     n_constr=0,
        #     xl=self.bounds[0,:],
        #     xu=self.bounds[1,:],
        # )
    
    def _evaluate_F(self, x):
        return self.Y
    
    def _evaluate_rho(self, x):
        return self.rho
    
    def get_domain(self):
        return self.bounds
    
    def f(self, X):
        return self.Y
        
    def get_noise_var(self, X):
        return self.rho


if __name__ == "__main__":
    
    prob = Exp()
    