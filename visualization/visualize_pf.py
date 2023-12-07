import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from arguments import get_vis_args
from utils import get_problem_dir, get_algo_names, defaultColorsCycle
import numpy as np


def get_data_of_step(pareto_approx_df, selected_iteration):
    filtered_data = pareto_approx_df[pareto_approx_df['iterID'] == selected_iteration]
    return filtered_data['Pareto_f1'], filtered_data['Pareto_f2']

def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    algo_names = get_algo_names(args)
    subfolder = args.subfolder

    n_algo, n_seed, seed = len(algo_names), args.n_seed, args.seed
    
    for i in range(n_algo):
        for j in range(n_seed):
            # Load data
            true_pareto_df = pd.read_csv(f'{problem_dir}/TrueParetoFront.csv')
            pareto_approx_df = pd.read_csv(f'{problem_dir}/{algo_names[i]}/{j}/ParetoFrontApproximation.csv')

            # Initial data for the plot

            # Create the figure
            fig = go.Figure()

            # Add traces
            fig.add_trace(go.Scatter(x=true_pareto_df['f1'], y=true_pareto_df['f2'], mode='markers', name='True Pareto Front'))
            # fig.add_trace(go.Scatter(x=initial_f1, y=initial_f2, mode='markers', name='Approximate Pareto Front'), secondary_y=False)

            # add all traces with loop
            for iteration in pareto_approx_df['iterID'].unique():
                
                f1, f2 = get_data_of_step(pareto_approx_df, iteration)
                f2 = f2
                fig.add_trace(go.Scatter(x=f1, y=f2, mode='markers', visible=False, name=f'Approximate Pareto Front at iteration {iteration}', marker_color=next(defaultColorsCycle)))
            
            fig.data[1].visible = True

            # Add slider
            steps = []
            for iteration in range(len(fig.data)):
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)},
                        {"title": f"Slider switched to iteration: {iteration}"}],  # layout attribute
                )
                step["args"][0]["visible"][iteration] = True
                step["args"][0]["visible"][0] = True  # show True Pareto Front at first
                steps.append(step)
                
            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Iteration: "},
                pad={"t": 10},
                steps=steps
            )]
            
            fig.update_layout(
                sliders=sliders
            )
            #fit layout to the real pareto front
            fig.update_layout(
            title="True vs Approximate Pareto Front",
            xaxis_title="Objective 1 (f1)",
            yaxis_title="Objective 2 (f2)",
            sliders=[{
                'active': 1,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16},
                    'prefix': 'Iteration:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
    }]
)
            
            # Show the plot
            if args.savefig:
                fig.write_image(f'{problem_dir}/{algo_names[i]}_seed{j}_pareto_front.html')
            else:
                fig.write_html(f'{problem_dir}/{algo_names[i]}_seed{j}_pareto_front.html')

if __name__ == '__main__':
    main()
