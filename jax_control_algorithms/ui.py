import IPython
import ipywidgets as widgets
from functools import partial

from jax_control_algorithms.trajectory_optimization import unpack_res


def solve_and_plot(solver, plot_fn):

    plot_output = widgets.Output()

    X_opt, U_opt, system_outputs, res = solver.run()
    is_converged, c_eq, c_ineq, trace, n_iter = unpack_res(res)

    with plot_output:
        IPython.display.clear_output(wait=True)  # Clear previous plot
        plot_fn(X_opt, U_opt, system_outputs, solver.problem_definition.parameters)

    return plot_output


def manual_investigate(solver, sliders, set_parameter_fn, plot_fn):

    # Create Output widgets for print outputs and plot
    print_output = widgets.Output()
    plot_output = widgets.Output()

    def update_plot(solver, **kwargs):

        X_opt, U_opt, system_outputs, res = None, None, None, None

        # compute
        with print_output:
            IPython.display.clear_output(wait=True)  # Clear previous print outputs

            set_parameter_fn(solver, **kwargs)
            X_opt, U_opt, system_outputs, res = solver.run()

        # unpack
        is_converged, c_eq, c_ineq, trace, n_iter = unpack_res(res)

        # show results
        with plot_output:
            IPython.display.clear_output(wait=True)  # Clear previous plot

            plot_fn(X_opt, U_opt, system_outputs, solver.problem_definition.parameters)

    ui = widgets.GridBox(list(sliders.values()), layout=widgets.Layout(grid_template_columns="repeat(3, 300px)"))

    interactive_plot = widgets.interactive_output(partial(update_plot, solver=solver), sliders)

    print_output_ = widgets.HBox([print_output], layout=widgets.Layout(
        height='350px',
        overflow_y='scroll',
    ))
    output_box = widgets.VBox([print_output_, plot_output])

    return ui, output_box, print_output, plot_output


def _plot_array_of_traces(X, n_iter, title_string='', figsize=(8, 5)):
    """
        Show plots that illustrate the convergence process of the solver.
    """

    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np

    def _get_color(i, i_max, colormap=cm.rainbow):
        c = float(i) / float(i_max)
        cindex = int(i_max * c)

        norm = plt.Normalize(vmin=0, vmax=1)
        sample_values = np.linspace(0, 1, i_max)
        rgba_samples = colormap(norm(sample_values))
        color = rgba_samples[cindex]

        return color

    n_lines = X.shape[2]

    fig, axes = plt.subplots(n_lines, 1, sharex=True, figsize=figsize, squeeze=False)

    for i_ax in range(n_lines):

        ax = axes[i_ax][0]

        lines = [ax.plot(X[i, :, i_ax], color=_get_color(i, n_iter))[0] for i in range(n_iter)]

        lines[0].set_label('first iteration i=0')
        lines[-1].set_label('last iteration i=' + str(n_iter - 1))
        ax.legend()
        ax.set_title(title_string + str(i_ax))

    return fig


def plot_iterations(res, figsize=(8, 5)):
    """
        Show plots that illustrate the convergence process of the solver given the solver results

        res - the solver results
        figsize - the size of the figure passed to matplotlib figure() function
    """
    is_converged, c_eq, c_ineq, traces, n_iter = unpack_res(res)

    fig1 = _plot_array_of_traces(traces['X_trace'], res.get('n_iter'), 'state ', figsize)
    fig2 = _plot_array_of_traces(traces['U_trace'], res.get('n_iter'), 'control variable ', figsize)

    return fig1, fig2
