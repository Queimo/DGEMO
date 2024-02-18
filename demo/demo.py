import numpy as np
import matplotlib.pyplot as plt

from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model

train_x = np.random.uniform(8, 20, 15) 
train_x_2 = train_x.copy()

train_y = 0.1 * train_x ** 2 - 3.5*train_x + np.cos(train_x*1.3) + 40 + np.random.normal(0, 0.01, len(train_x)) * train_x**2 * 0.3
train_y_2 = 0.1 * train_x_2 ** 2 - 3.5*train_x_2 + np.cos(train_x_2*1.3) + 40 + np.random.normal(0, 0.01, len(train_x_2)) * train_x_2**2 * 0.3

train_x = np.stack([train_x, train_x_2], axis=1)
train_y = np.stack([train_y, train_y_2], axis=1)

# Create some "gap" in the data
# to test epistemic uncertainty
# estimation:
# Create masks for each condition
mask1 = (train_x[:, 0] >= 12) & (train_x[:, 0] <= 16)
mask2 = (train_x[:, 1] >= 12) & (train_x[:, 1] <= 16)

# Combine the masks with logical AND
combined_mask = mask1 & mask2

# Use the combined mask to filter train_x and train_y
train_x = train_x[combined_mask]
train_y = train_y[combined_mask]


# Scaling:
train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
train_y = (train_y - train_y.mean()) / train_y.std()

plt.plot(train_x, train_y, 'k*')

import torch
import gpytorch
import pyro

class HeteroskedasticGP_pyro(gpytorch.models.ApproximateGP):
    
    def __init__(self, num_inducing=64, name_prefix="heteroskedastic_gp"):
        self.name_prefix = name_prefix

        # Define all the variational stuff for 2D inputs
        inducing_points = torch.rand(2, num_inducing, 2)  # 2D inducing points for 2 tasks
        
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, 
            batch_shape=torch.Size([2]))

        single_variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True)

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                single_variational_strategy,
                num_tasks=2,
                num_latents=2)

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def guide(self, x, y):
        function_dist = self.pyro_guide(x)
        with pyro.plate(self.name_prefix + ".data_plate", dim=-2):
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp", self)
        function_dist = self.pyro_model(x)
        with pyro.plate(self.name_prefix + ".data_plate", dim=-2):
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)
            mean_samples = function_samples[..., 0]
            std_samples = function_samples[..., 1]
            transformed_std_samples = torch.exp(std_samples)
            return pyro.sample(self.name_prefix + ".y",
                               pyro.distributions.Normal(mean_samples, transformed_std_samples),
                               obs=y)

def initialize_model(train_x, train_y, train_rho=None):
    # define models for objective and constraint
    train_y_mean = -train_y  # negative because botorch assumes maximization
    # train_y_var = self.real_problem.evaluate(train_x).to(**tkwargs).var(dim=-1)
    train_y_var = train_rho + 1e-6
    models = []
    for i in range(train_y.shape[1]):
        train_y_i = train_y_mean[..., i]
        train_yvar_i = train_y_var[..., i]
        # models.append(
        #     FixedNoiseGP(
        #         train_X=train_x,
        #         train_Y=train_y_i.unsqueeze(-1),
        #         train_Yvar=train_yvar_i.unsqueeze(-1),
        #         outcome_transform=Standardize(m=1),
        #     )
        # )
        models.append(
            HeteroskedasticGP_pyro(
                train_X=train_x,
                train_Y=train_y_i.unsqueeze(-1),
                train_Yvar=train_yvar_i.unsqueeze(-1),
            )
        )
        # models.append(
        #     SingleTaskGP(
        #         train_x,
        #         train_y_i.unsqueeze(-1),
        #         # train_yvar_i.unsqueeze(-1),
        #         outcome_transform=Standardize(m=1),
        #     )
        # )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    return mll, model

# Instantiate model:
pyro.clear_param_store()  # Good practice
model1 = HeteroskedasticGP_pyro()
model2 = HeteroskedasticGP_pyro()

model = HeteroskedasticGP_pyro()



num_iter = 1000
num_particles = 256


optimizer = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True)
svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)
model.train()
iterator = range(num_iter)
for i in iterator:
    model.zero_grad()
    loss = svi.step(torch.from_numpy(train_x).float().unsqueeze(1), torch.from_numpy(train_y).float())
    print(model, loss)

# training routine:
def train():
    for i,model in enumerate([model1, model2]):
        optimizer = pyro.optim.Adam({"lr": 0.01})
        elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True)
        svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

        model.train()
        iterator = range(num_iter)
        for i in iterator:
            model.zero_grad()
            loss = svi.step(torch.from_numpy(train_x).float().unsqueeze(1), torch.from_numpy(train_y[:,i]).float())
            print(model, loss)

# Train the GP:
train()

# Function to plot predictions:
def plot_preds(ax, feature, mean_samples, var_samples, bootstrap=True, n_boots=100,
               show_epistemic=False, epistemic_mean=None, epistemic_var=None):
    """Plots the overall mean and variance of the aggregate system.
    Inherited from https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Heteroskedastic.html.

    We can represent the overall uncertainty via explicitly sampling the underlying normal
    distributrions (with `bootstrap=True`) or as the mean +/- the standard deviation from
    the Law of Total Variance. 
    
    For systems with many observations, there will likely be
    little difference, but in cases with few observations and informative priors, plotting
    the percentiles will likely give a more accurate representation.
    """
    if bootstrap:
        means = np.expand_dims(mean_samples.T, axis=2)
        stds = np.sqrt(np.expand_dims(var_samples.T, axis=2))
        
        samples_shape = mean_samples.T.shape + (n_boots,)
        samples = np.random.normal(means, stds, samples_shape)
        
        reshaped_samples = samples.reshape(mean_samples.shape[1], -1).T
        
        l, m, u = [np.percentile(reshaped_samples, p, axis=0) for p in [2.5, 50, 97.5]]
        ax.plot(feature, m, label="Median", color="b")
    
    else:
        m = mean_samples.mean(axis=0)
        sd = np.sqrt(mean_samples.var(axis=0) + var_samples.mean(axis=0))
        l, u = m - 1.96 * sd, m + 1.96 * sd
        ax.plot(feature, m, label="Mean", color="b")
    
    if show_epistemic:
        ax.fill_between(feature,
                        l,
                        epistemic_mean-1.96*np.sqrt(epistemic_var), 
                        alpha=0.2,
                        color="#88b4d2",
                        label="Total Uncertainty (95%)")
        ax.fill_between(feature, 
                        u, 
                        epistemic_mean+1.96*np.sqrt(epistemic_var), 
                        alpha=0.2,
                        color="#88b4d2")
        ax.fill_between(feature, 
                        epistemic_mean-1.96*np.sqrt(epistemic_var), 
                        epistemic_mean+1.96*np.sqrt(epistemic_var),
                        alpha=0.4,
                        color="#88b4d2",
                        label="Epistemic Uncertainty (95%)")
    else:
        ax.fill_between(feature, 
                        l, 
                        u, 
                        alpha=0.2,
                        color="#88b4d2",
                        label="Total Uncertainty (95%)")

# Test data to predict:
x_padding = 0.1

test_x = torch.linspace(train_x.min() - (train_x.max() - train_x.min()) * x_padding, 
                        train_x.max() + (train_x.max() - train_x.min()) * x_padding, 
                        100).float()

# Predict:
# model.eval()
# with torch.no_grad():
#     output_dist = model(test_x)

# # Extract predictions:
# output_samples = output_dist.sample(torch.Size([1000]))
# mu_samples = output_samples[...,0]
# sigma_samples = torch.exp(output_samples[...,1])

# # Plot predictions:
# plt.plot(train_x, train_y, 'k*', label="Observed Data")
# ax = plt.gca()
# plot_preds(ax, 
#            test_x.numpy(),
#            mu_samples.numpy(), 
#            sigma_samples.numpy()**2,
#            bootstrap=False, 
#            n_boots=100,
#            show_epistemic=True,
#            epistemic_mean=output_dist.mean[:,0].detach().numpy(),
#            epistemic_var=output_dist.stddev[:,0].detach().numpy()**2)
# ax.legend();
