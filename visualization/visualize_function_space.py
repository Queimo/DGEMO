# plot each iteration of predicted Pareto front, proposed points, evaluated points for two algorithms
import pathlib
import os, sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from arguments import get_vis_args
from utils import get_problem_dir, get_algo_names, defaultColors
import yaml


def plotly_grid_plotter(figures=[], path="grid_plots.html", ncols=3):
    # Start HTML string with doctype and head including CSS styles
    html_string = (
        """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <title>Plotly Plots</title>
    <style>
    /* Add CSS to style the plot containers */
    .grid-container {
    display: grid;
    """
        + f"grid-template-columns: repeat({ncols}, 1fr);"
        + """
    grid-gap: 10px; /* space between plots */
    }
    .grid-item {
    margin: 10px;
    }
    </style>
    </head>
    <body>
    <div class="grid-container">
    """
    )

    # Add the HTML for each figure wrapped in div.grid-item
    for fig in figures:
        fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        html_string += f'<div class="grid-item">{fig_html}</div>'

    # Close div.grid-container and body/html tags
    html_string += """
    </div>
    </body>
    </html>
    """

    # Write the HTML to a file
    with open(path, "w") as file:
        file.write(html_string)


def get_data_of_step(pareto_approx_df, selected_iteration):
    filtered_data = pareto_approx_df[pareto_approx_df["iterID"] == selected_iteration]
    return filtered_data


def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    algo_names = get_algo_names(args)
    seed = args.seed

    n_algo = len(algo_names)
    problem_name = os.path.basename(os.path.dirname(problem_dir))

    # read result csvs
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    algo_names = get_algo_names(args)
    seed = args.seed

    n_algo = len(algo_names)
    problem_name = os.path.basename(os.path.dirname(problem_dir))

    # read result csvs
    data_list, paretoEval_list, paretoGP_list, yml_list = [], [], [], []
    for algo_name in algo_names:
        csv_folder = f"{problem_dir}/{algo_name}/{seed}/"
        data_list.append(pd.read_csv(csv_folder + "EvaluatedSamples.csv"))
        paretoEval_list.append(pd.read_csv(csv_folder + "ParetoFrontEvaluated.csv"))
        with open(csv_folder + "args.yml") as f:
            yml_list.append(yaml.load(f, Loader=yaml.SafeLoader))
        paretoGP_list.append(pd.read_csv(csv_folder + "ParetoFrontApproximation.csv"))

    true_front_file = os.path.join(problem_dir, "TrueParetoFront0.csv")
    has_true_front = os.path.exists(true_front_file)
    if has_true_front:
        df_truefront = pd.read_csv(true_front_file)

    # get all true front files
    tf_paths = pathlib.Path(problem_dir).glob("TrueParetoFront*.csv")
    df_truefront_list = [pd.read_csv(str(tf_path)) for tf_path in tf_paths]

    n_var = len(
        [key for key in data_list[0] if len(key) == 1 and key <= "Z" and key >= "A"]
    )
    n_obj = len([key for key in data_list[0] if key.startswith("f")])



    algo = algo_names[0]

    # Create one figure for each seed
    fig = go.Figure()

    approx_all_df = pd.read_csv(f"{problem_dir}/{algo}/{seed}/ApproximationAll.csv")

    # label the sample
    def makeLabel(dfRow):
        retStr = "Data:<br>"
        for col in dfRow.index:  # Iterate over all columns
            retStr += f"{col}: {round(dfRow[col], 2)}<br>"
        return retStr

    # Set hovertext (label)
    for df in data_list + paretoEval_list + paretoGP_list + [approx_all_df]:
        df["hovertext"] = df.apply(makeLabel, axis=1)
    # Maximum number of iterations to display
    
    max_iterations = approx_all_df["iterID"].unique().shape[0] +1

    # get one iteration to check length of data and take square root
    approx_all_i = get_data_of_step(approx_all_df, 1)
    n_grid = int(np.sqrt(approx_all_i.shape[0]))

    kk = 0

    # Add algorithm traces for each iteration
    for iteration in range(1,max_iterations):
        approx_all_i = get_data_of_step(approx_all_df, iteration)
        # Trimming our DataFrames to the matching iterID
        data_trimmed = data_list[kk][data_list[kk]["iterID"] < iteration]
        last_eval = iteration
        # Getting Data of last evaluated points points
        data_lastevaluated = data_list[kk][data_list[kk]["iterID"] == last_eval]
        # Getting Data of proposed points
        data_proposed = data_list[kk][data_list[kk]["iterID"] == last_eval]
        # First set of samples
        firstsamples = data_list[kk][data_list[kk]["iterID"] == 0]
        paretoEval_trimmed = paretoEval_list[kk][
            paretoEval_list[kk]["iterID"] == iteration
        ]
        paretoGP_trimmed = paretoGP_list[kk][paretoGP_list[kk]["iterID"] == iteration]

        fig.add_trace(
            go.Surface(
                x=approx_all_i["x1"].values.reshape((n_grid, n_grid)),
                y=approx_all_i["x2"].values.reshape((n_grid, n_grid)),
                z=approx_all_i["F_1"].values.reshape((n_grid, n_grid)),
                opacity=0.8,
                visible=(iteration == 1),
                surfacecolor=approx_all_i["S_1"].values.reshape((n_grid, n_grid)),
                # add color
                colorscale="Viridis",
            )
        )

        fig.add_trace(
            go.Surface(
                x=approx_all_i["x1"].values.reshape((n_grid, n_grid)),
                y=approx_all_i["x2"].values.reshape((n_grid, n_grid)),
                z=approx_all_i["F_2"].values.reshape((n_grid, n_grid)),
                opacity=0.8,
                visible=(iteration == 1),
                surfacecolor=approx_all_i["S_2"].values.reshape((n_grid, n_grid)),
                # add color
                # colorscale='Viridis'
            )
        )

        # add evaluation points
        fig.add_trace(
            go.Scatter3d(
                x=data_trimmed["x1"],
                y=data_trimmed["x2"],
                z=data_trimmed["f1"],
                mode="markers",
                visible=(iteration == 0),
                marker=dict(
                    size=3, color="red", symbol="x", line=dict(color="green", width=6)
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=data_trimmed["x1"],
                y=data_trimmed["x2"],
                z=data_trimmed["f2"],
                mode="markers",
                visible=(iteration == 0),
                marker=dict(
                    size=3, color="red", symbol="x", line=dict(color="red", width=6)
                ),
            )
        )

        # data proposed
        fig.add_trace(
            go.Scatter3d(
                x=data_proposed["x1"],
                y=data_proposed["x2"],
                z=data_proposed["f1"],
                mode="markers",
                visible=(iteration == 0),
                marker=dict(
                    size=8,
                    color="green",
                    symbol="circle",
                    line=dict(color="red", width=6),
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=data_proposed["x1"],
                y=data_proposed["x2"],
                z=data_proposed["f2"],
                mode="markers",
                visible=(iteration == 0),
                marker=dict(
                    size=8,
                    color="green",
                    symbol="circle",
                    line=dict(color="pink", width=6),
                ),
            )
        )

        # fig.add_trace(go.Scatter(x=f1, y=f2, mode='markers', visible=(iteration==0),
        #                             name=f'{algo_names[i]} - Iter {iteration}',
        #                             marker_color=defaultColors[i],
        #                             opacity=0.5,
        #                             ))

    # Create and add slider
    steps = []
    for iteration in range(1, max_iterations):
        iteration = dict(
            method="update",
            args=[
                {
                    "visible": [
                        iteration == i
                        for i in range(1, max_iterations)
                        for j in range(int(len(fig.data) / (max_iterations - 1)))
                    ]
                },
                {"title": f"Slider switched to iteration: {iteration}"},
            ],
        )
        steps.append(iteration)

    sliders = [
        dict(
            active=0, currentvalue={"prefix": "Iteration: "}, pad={"t": 10}, steps=steps
        )
    ]

    fig.update_layout(sliders=sliders)
    # Update layout
    fig.update_layout(
        title="Surface Plot",
        title_font_size=20,
        title_x=0.5,  # Center title
        font=dict(
            family="Arial, sans-serif", size=12, color="black"
        ),  # Professional font
        scene=dict(
            xaxis_title="X Axis Title",  # Replace with your title
            yaxis_title="Y Axis Title",  # Replace with your title
            zaxis_title="Z Axis Title",  # Replace with your title
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgrey"),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgrey"),
            zaxis=dict(
                showgrid=True, gridwidth=1, gridcolor="lightgrey", range=[0, 1.5]
            ),
        ),
        margin=dict(l=65, r=50, b=65, t=90),
    )
    # axis labels
    fig.update_layout(
        scene=dict(
            yaxis_title="ZnSO4", xaxis_title="NaOH/ZnSO4", zaxis_title="(pH - 9)^2"
        ),
        title="pH",
        autosize=False,
        width=1500,
        height=1500,
        # cube axes
        # scene_aspectmode='cube',
    )
    fig.show()

    # Assuming 'fig' is a previously initialized figure object
    # Initialize with subplots for contour plots

    rows = 2
    cols = 3

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=("F_i", "S_i", "rho_F_i")
    )

    for iteration in range(1, max_iterations):
        approx_all_i = get_data_of_step(approx_all_df, iteration)
        # Trimming our DataFrames to the matching iterID
        data_trimmed = data_list[kk][data_list[kk]["iterID"] < iteration]
        last_eval = iteration
        # Getting Data of last evaluated points points
        data_lastevaluated = data_list[kk][data_list[kk]["iterID"] == last_eval]
        # Getting Data of proposed points
        data_proposed = data_list[kk][data_list[kk]["iterID"] == last_eval]
        # First set of samples
        firstsamples = data_list[kk][data_list[kk]["iterID"] == 0]
        paretoEval_trimmed = paretoEval_list[kk][
            paretoEval_list[kk]["iterID"] == iteration
        ]
        paretoGP_trimmed = paretoGP_list[kk][paretoGP_list[kk]["iterID"] == iteration]

        # Data reshaping remains the same
        x = approx_all_i["x1"].values.reshape((n_grid, n_grid))
        y = approx_all_i["x2"].values.reshape((n_grid, n_grid))

        for i in range(1, rows + 1):
            fig.add_trace(
                go.Contour(
                    x=x[0],
                    y=y[:, 0],
                    z=approx_all_i[f"F_{i}"].values.reshape((n_grid, n_grid)),
                    colorscale="Viridis",
                    showscale=True,
                    visible=(iteration == 1),
                    zmin=min(approx_all_df[f"F_{i}"]),
                    zmax=max(approx_all_df[f"F_{i}"])
                ),
                row=i,
                col=1,
            )

            # add S_1 S_2

            fig.add_trace(
                go.Contour(
                    x=x[0],
                    y=y[:, 0],
                    z=approx_all_i[f"S_{i}"].values.reshape((n_grid, n_grid)),
                    colorscale="Viridis",
                    showscale=True,
                    visible=(iteration == 1),
                    zmin=min(approx_all_df[f"S_{i}"]),
                    zmax=max(approx_all_df[f"S_{i}"])
                ),
                row=i,
                col=2,
            )
            
            # add rho_1 rho_2
            fig.add_trace(
                go.Contour(
                    x=x[0],
                    y=y[:, 0],
                    z=approx_all_i[f"rho_F_{i}"].values.reshape((n_grid, n_grid)),
                    colorscale="Viridis",
                    showscale=True,
                    visible=(iteration == 1),
                    zmin=min(approx_all_df[f"rho_F_{i}"]),
                    zmax=max(approx_all_df[f"rho_F_{i}"])*0.8
                ),
                row=i,
                col=3,
            )

        # data proposed
        for i in range(1, cols + 1):
            for j in range(1, rows + 1):
                fig.add_trace(
                    go.Scatter(
                        x=data_trimmed["x1"],
                        y=data_trimmed["x2"],
                        mode="markers",
                        hovertext=data_trimmed['hovertext'],
                        hoverinfo="text",
                        visible=(iteration == 1),
                        marker=dict(
                            size=8,
                            color="grey",
                            symbol="circle",
                            line=dict(color="black", width=1),
                        ),
                    ),
                    row=j,
                    col=i,
                )

                # data proposed
                fig.add_trace(
                    go.Scatter(
                        x=data_proposed["x1"],
                        y=data_proposed["x2"],
                        mode="markers",
                        hovertext=data_proposed['hovertext'],
                        hoverinfo="text",
                        visible=(iteration == 1),
                        marker=dict(
                            size=15,
                            color="red",
                            symbol="x",
                            line=dict(color="black", width=1),
                        ),
                    ),
                    row=j,
                    col=i,
                )

                # add evaluation pareto front with yellow squares
                fig.add_trace(
                    go.Scatter(
                        x=paretoEval_trimmed["x1"],
                        y=paretoEval_trimmed["x2"],
                        mode="markers",
                        visible=(iteration == 1),
                        marker=dict(
                            size=8,
                            color="yellow",
                            symbol="square",
                            line=dict(color="black", width=1),
                        ),
                    ),
                    row=j,
                    col=i,
                )
                
                #paretoGP_trimmed low opacity orange circles
                 
                 
                fig.add_trace(
                    go.Scatter(
                        x=paretoGP_trimmed["x1"],
                        y=paretoGP_trimmed["x2"],
                        mode="markers",
                        visible=(iteration == 1),
                        marker=dict(
                            size=4,
                            color="orange",
                            symbol="circle",
                            opacity=0.9
                        ),
                    ),
                    row=j,
                    col=i,
                )
                           
    # Slider setup (similar to your original setup)
    steps = []
    for iteration in range(1, max_iterations):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [
                        iteration == i
                        for i in range(1, max_iterations)
                        for j in range(int(len(fig.data) / (max_iterations - 1)))
                    ]
                },
                {"title": f"Slider switched to iteration: {iteration}"},
            ],
        )
        steps.append(step)

    sliders = [
        dict(
            active=0, currentvalue={"prefix": "Iteration: "}, pad={"t": 50}, steps=steps
        )
    ]

    fig.update_layout(sliders=sliders)

    fig.show()

    # # Show or save the plot
    # plotly_grid_plotter(fig, f'./result/{args.problem}/{args.subfolder}/{args.problem}_seed{seed}_performance_space.html', ncols=2 if n_algo > 1 else 1)
    if args.savefig:
        fig.write_image(f'{problem_dir}/seed{j}_{algo}_IO_space.png')
    else:
        fig.write_html(f'{problem_dir}/seed{j}_{algo}_IO_space.html')

    print(f"Saved {problem_dir}/seed{j}_{algo}_IO_space")


if __name__ == "__main__":
    main()
