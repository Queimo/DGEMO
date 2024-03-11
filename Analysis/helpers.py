import torch
import pandas as pd
from pathlib import Path
import numpy as np
import yaml


def load_from_path(path):

    eval_samples = pd.read_csv(path / "EvaluatedSamples.csv")
    approx_all_df = pd.read_csv(path / "ApproximationAll.csv")
    paretoGP = pd.read_csv(path / "ParetoFrontApproximation.csv")
    paretoEval = pd.read_csv(path / "ParetoFrontEvaluated.csv")
    paretoEvalMVaR = pd.read_csv(path / "MVaRParetoFrontEvaluated.csv")
    paretoApprox = pd.read_csv(path / "ParetoFrontApproximation.csv")
    
    eval_samples = pd.read_csv(path / "EvaluatedSamples.csv")
    eval_samples.columns.to_list()

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

def save_pdf(plt, path, name, idx=""):
    path_folder = Path("./Plots") / f"{name}"
    path_folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(path_folder / ("_".join(path.parts[2:-1]) + f"{idx}.pdf"), dpi=300, format="pdf")