import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from argparse import ArgumentParser
from problems.common import build_problem

import torch
from botorch.utils.multi_objective.hypervolume import infer_reference_point


class RefPoint:
    
    ref_point_botroch = None
    ref_point_pymoo = None
    
    def __init__(self, problem, n_var=6, n_obj=2, n_init_sample=100, seed=0, is_botorch=False):
        
        np.random.seed(seed)
        _, _, _, Y_init = build_problem(problem, n_var, n_obj, n_init_sample)
    
        # self.ref_point_botroch = infer_reference_point(torch.tensor(-Y_init)).numpy().tolist()
        self.ref_point_botroch = np.max(-Y_init, axis=0).tolist()
        self.ref_point_pymoo= np.max(Y_init, axis=0).tolist()
        # self.ref_point_pymoo = (-infer_reference_point(torch.tensor(-Y_init)).numpy()).tolist()
        print(self)

    def get_ref_point(self, is_botorch=False):

        if is_botorch:
            return self.ref_point_botroch
        else:
            return self.ref_point_pymoo
        
    def __str__(self):
        return f'Ref point: {self.ref_point_pymoo} \n Ref point (botorch): {self.ref_point_botroch}'
    
    def __repr__(self): 
        return self.__str__()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--n-var', type=int, default=2)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--n-init-sample', type=int, default=500)
    args = parser.parse_args()

    ref_point_handler = RefPoint(args.problem, args.n_var, args.n_obj, args.n_init_sample)

    print(ref_point_handler)