import torch
import pandas as pd
from pathlib import Path
import numpy as np
import yaml

from scipy.stats import norm

def calculate_var_pos(mean, std_dev, alpha=0.9):

    # Calculate the z-score for the given alpha level
    z_score = norm.ppf(alpha)

    # Calculate mVaR for each variable
    var = mean - z_score * std_dev
    return var


def get_dfs(exp):
    df = pd.read_excel(f"./final/{exp}/XRD+synthsis_data.xlsx")
    path = Path(f"./final/{exp}/unroll_new/0/")
    # load the data
    res_dict = load_from_path(path)
    eval_samples = res_dict["eval_samples"]
    eval_samples["id"] = eval_samples.index
    # mvar_hv = eval_samples["MVaR_Hypervolume_indicator"]

    df.dropna(how="any", axis=1, inplace=True)

    # merge eval_samples onto df on id
    df = pd.merge(df, eval_samples, on="id")

    df.reset_index(drop=True, inplace=True)

    df_mean = df.select_dtypes(include=["float64", "int64"]).groupby("id").mean()
    df_std = df.select_dtypes(include=["float64", "int64"]).groupby("id").std()

    # only std non-zero columns
    df_std = df_std.loc[:, df_std.mean() > 0]

    df_mean_std = pd.merge(
        df_mean, df_std, left_index=True, right_index=True, suffixes=("", "_std")
    )

    for col in df_std.columns:
        df_mean_std[col + "_VaR"] = calculate_var_pos(
            mean=df_mean[col],
            std_dev=df_std[col]
        )
        df_mean_std.rename(columns={col: col + "_mean"}, inplace=True)


    df_mean_std["id"] = df_mean_std.index

    df_mean_std["initial_sampling"] = df_mean_std["id"] < 12
    df["initial_sampling"] = df["id"] < 12
    df_mean_std["zero"] = 0

    # reset index
    df_mean_std.reset_index(drop=True, inplace=True)
    
    return df, df_mean_std

def load_from_path(path):

    eval_samples = pd.read_csv(path / "EvaluatedSamples.csv")
    approx_all_df = pd.read_csv(path / "ApproximationAll.csv")
    paretoGP = pd.read_csv(path / "ParetoFrontApproximation.csv")
    paretoEval = pd.read_csv(path / "ParetoFrontEvaluated.csv")
    paretoEvalMVaR = pd.read_csv(path / "MVaRParetoFrontEvaluated.csv")
    paretoApprox = pd.read_csv(path / "ParetoFrontApproximation.csv")

    args_yaml = path / "args.yml"
    args = yaml.safe_load(args_yaml.open("r"))

    batch_size = args["general"]["batch_size"]
    init_samples = args["general"]["n_init_sample"]
    n_obj = args["general"]["n_obj"]

    sub=0
    algo_name = args["general"]["algo"]
    if "det" in algo_name:
        sub=1

    if "exp" in args["general"]["problem"]:
        eval_samples = eval_samples[~(eval_samples["iterID"] == eval_samples["iterID"].max())]
        sub = 1
        
    n_obj -= sub
        
    # bool array for sobol experiments vs optimization
    sobol = np.where(eval_samples["iterID"] == 0, True, False)
    res_dict = {
        "eval_samples": eval_samples,
        "approx_all_df": approx_all_df,
        "paretoEval": paretoEval,
        "paretoEvalMVaR": paretoEvalMVaR,
        "paretoApprox": paretoApprox,
        "paretoGP": paretoGP,
        "args": args,
        "sobol": sobol,
        "batch_size": batch_size,
        "init_samples": init_samples,
        "n_obj": n_obj,
    }
    return res_dict

def save_image(plt, path, name, idx="", format="pdf"):
    path_folder = Path("./Plots") / f"{name}"
    path_folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(path_folder / ("_".join(path.parts[2:-1]) + f"{idx}.{format}"), dpi=300, format=format)