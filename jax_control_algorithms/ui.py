import math
#import matplotlib.pyplot as plt 
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
        plot_fn(X_opt, U_opt, system_outputs, solver.theta)

    return plot_output

def manual_investigate(solver, sliders, set_theta_fn, plot_fn):
    
    # Create Output widgets for print outputs and plot
    print_output = widgets.Output()
    plot_output = widgets.Output()

    def update_plot(solver, **kwargs):
        
        X_opt, U_opt, system_outputs, res = None, None, None, None

        # compute
        with print_output:
            IPython.display.clear_output(wait=True)  # Clear previous print outputs

            set_theta_fn(solver, **kwargs)
            X_opt, U_opt, system_outputs, res = solver.run()
        
        
        # unpack
        is_converged, c_eq, c_ineq, trace, n_iter = unpack_res(res)

        # show results
        with plot_output:
            IPython.display.clear_output(wait=True)  # Clear previous plot

            plot_fn(X_opt, U_opt, system_outputs, solver.theta)


    ui = widgets.GridBox(list( sliders.values() ), layout=widgets.Layout(grid_template_columns="repeat(3, 300px)"))

    interactive_plot = widgets.interactive_output(partial(update_plot, solver=solver), sliders)

    print_output_ = widgets.HBox([print_output], layout=widgets.Layout(height='350px', overflow_y='scroll', ))
    output_box = widgets.VBox([print_output_, plot_output])

    return ui, output_box, print_output, plot_output