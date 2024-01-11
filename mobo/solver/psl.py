import numpy as np
from .solver import Solver

import numpy as np
import torch


from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

"""
A simple FC Pareto Set model.
"""

import torch
import torch.nn as nn

class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_var, n_obj):
        super(ParetoSetModel, self).__init__()
        self.n_var = n_var
        self.n_obj = n_obj
       
        self.fc1 = nn.Linear(self.n_obj, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_var)
       
    def forward(self, pref):

        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = torch.sigmoid(x) 
        
        return x.to(torch.float64)


# PSL 
# number of learning steps
n_steps = 1000 
# number of sampled preferences per step
n_pref_update = 10 
# coefficient of LCB
coef_lcb = 0.1
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 128 
# number of optional local search
n_local = 0
# device
device = 'cpu'



class PSLSolver(Solver):
    '''
    Solver based on PSL
    '''
    def __init__(self, *args, **kwargs):
        self.z = torch.zeros(kwargs["n_obj"]).to(device)
        self.psmodel = None
        super().__init__(algo="",*args,**kwargs)
        
    def save_psmodel(self, path):
        torch.save(self.psmodel.state_dict(), f'{path}/psmodel.pt')

    def solve(self, problem, X, Y, rho=None):
        
        surrogate_model = problem.surrogate_model
        
        self.z =  torch.min(torch.cat((self.z.reshape(1,surrogate_model.n_obj),torch.from_numpy(Y).to(device) - 0.1)), axis = 0).values.data
        
        # nondominated X, Y 
        nds = NonDominatedSorting()
        idx_nds = nds.do(Y)
        
        X_nds = X[idx_nds[0]]
        Y_nds = Y[idx_nds[0]]
            
        # intitialize the model and optimizer 
        self.psmodel = ParetoSetModel(surrogate_model.n_var, surrogate_model.n_obj)
        self.psmodel.to(device)
            
        # optimizer
        optimizer = torch.optim.Adam(self.psmodel.parameters(), lr=1e-3)
          
        # t_step Pareto Set Learning with Gaussian Process
        self.psmodel.train()
        for t_step in range(n_steps):
            
            # sample n_pref_update preferences
            alpha = np.ones(surrogate_model.n_obj)
            pref = np.random.dirichlet(alpha,n_pref_update)
            pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
            
            # get the current coressponding solutions
            x = self.psmodel(pref_vec)
            x_np = x.detach().cpu().numpy()
            
            # obtain the value/grad of mean/std for each obj
            # TODO only call once
            mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
            mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)
            
            std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
            std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).to(device)
            
            # calculate the value/grad of tch decomposition with LCB
            value = mean - coef_lcb * std
            value_grad = mean_grad - coef_lcb * std_grad
            
            tch_idx = torch.argmax((1 / pref_vec) * (value - self.z), axis = 1)
            tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
            tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update,1) *  value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis = 1) 

            tch_grad = tch_grad / torch.norm(tch_grad, dim = 1)[:, None]
            
            # gradient-based pareto set model update 
            optimizer.zero_grad()
            self.psmodel(pref_vec).backward(tch_grad)
            optimizer.step()  
            
        # solutions selection on the learned Pareto set
        self.psmodel.eval()
        
        # sample n_candidate preferences
        alpha = np.ones(surrogate_model.n_obj)
        pref = np.random.dirichlet(alpha,n_candidate)
        pref  = torch.tensor(pref).to(device).float() + 0.0001

        # generate correponding solutions, get the predicted mean/std
        X_candidate = self.psmodel(pref).to(torch.float64)
        X_candidate_np = X_candidate.detach().cpu().numpy()
        Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
        
        Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
        Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
        
        # optional TCH-based local Exploitation 
        # if n_local > 0:
        #     X_candidate_tch = X_candidate_np
        #     z_candidate = self.z.cpu().numpy()
        #     pref_np = pref.cpu().numpy()
        #     for _ in range(n_local):
        #         candidate_mean =  surrogate_model.evaluate(X_candidate_tch)['F']
        #         candidate_mean_grad =  surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']
                
        #         candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
        #         candidate_std_grad = surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']
                
        #         candidate_value = candidate_mean - coef_lcb * candidate_std
        #         candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad
                
        #         candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis = 1)
        #         candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)]
                
        #         candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)].reshape(n_candidate,1) * candidate_grad[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)] 
        #         candidate_tch_grad +=  0.01 * np.sum(candidate_grad, axis = 1) 
                
        #         X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
        #         X_candidate_tch[X_candidate_tch <= 0]  = 0
        #         X_candidate_tch[X_candidate_tch >= 1]  = 1  
                
        #     X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])
            
        #     Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
        #     Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
            
        #     Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
        
        
        # construct solution
        self.solution = {'x': np.array(X_candidate_np), 'y': np.array(Y_candidate)}
        return self.solution