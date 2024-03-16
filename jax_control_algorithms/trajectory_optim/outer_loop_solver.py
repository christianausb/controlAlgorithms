import jax
import jax.numpy as jnp
import jaxopt

#from jax_control_algorithms.common import *
from jax_control_algorithms.jax_helper import *
"""

"""


def _run_outer_loop(
    i, variables, parameters_of_dynamic_model, penality_parameter_trace, opt_c_eq, verification_state_init, solver_settings,
    objective_fn, verification_fn, verbose, print_errors, target_dtype
):
    """
        Execute the outer loop of the optimization process: herein in each iteration, the parameters of the
        boundary function and the quality cost weight factor are adjusted so that in the final iteration 
        the equality and inequality constraints are fulfilled.
    """

    # convert dtypes
    (
        variables,
        parameters_of_dynamic_model,
        penality_parameter_trace,
        opt_c_eq,
        verification_state_init,
        lam,
        tol_inner,
    ) = convert_dtype(
        (
            variables,
            parameters_of_dynamic_model,
            penality_parameter_trace,
            opt_c_eq,
            verification_state_init,
            solver_settings['lam'],
            solver_settings['tol_inner'],
        ), target_dtype
    )

    # _solver_settings = convert_dtype(solver_settings, target_dtype)

    #
    # loop:
    #

    def loop_body(loop_par):

        # loop iteration variable i
        i = loop_par['i']

        # get the penality parameter
        penality_parameter = loop_par['penality_parameter_trace'][i]
        is_finished_2 = i >= loop_par['penality_parameter_trace'].shape[0] - 1

        #
        parameters_passed_to_inner_solver = loop_par['parameters_of_dynamic_model'] + (
            penality_parameter,
            loop_par['opt_c_eq'],
        )

        # run inner solver
        gd = jaxopt.BFGS(
            fun=objective_fn, value_and_grad=False, tol=loop_par['tol_inner'], maxiter=solver_settings['max_iter_inner']
        )
        res = gd.run(loop_par['variables'], parameters=parameters_passed_to_inner_solver)
        _variables_next = res.params

        # run callback to verify the solution
        verification_state_next, is_finished_1, is_eq_converged, is_abort, is_X_finite, i_best = verification_fn(
            loop_par['verification_state'], i, res, _variables_next, loop_par['parameters_of_dynamic_model'], penality_parameter
        )

        # c_eq-control
        opt_c_eq_next = jnp.where(
            is_eq_converged,

            # in case of convergence of the error below the threshold there is not need to increase c_eq
            loop_par['opt_c_eq'],

            # increase c_eq
            loop_par['opt_c_eq'] * lam,
        )

        # use previous state of the iteration in case of abortion (when is_abort == True)
        variables_next = (
            jnp.where(
                is_abort,
                loop_par['variables'][0],
                _variables_next[0]  #
            ),
            jnp.where(
                is_abort,
                loop_par['variables'][1],  # use previous state of the iteration in case of abortion
                _variables_next[1]  #
            ),
        )

        # solution found?
        is_finished = jnp.logical_and(is_finished_1, is_finished_2)

        if verbose:
            lax.cond(is_finished, lambda: jax.debug.print("✅ found feasible solution"), lambda: None)

        loop_par = {
            'is_finished': is_finished,
            'is_abort': is_abort,
            'is_X_finite': is_X_finite,
            'variables': variables_next,
            'parameters_of_dynamic_model': loop_par['parameters_of_dynamic_model'],
            'penality_parameter_trace': penality_parameter_trace,
            'opt_c_eq': opt_c_eq_next,
            'i': loop_par['i'] + 1,
            'verification_state': verification_state_next,
            'tol_inner': loop_par['tol_inner'],
        }

        return loop_par

    def loop_cond(loop_par):
        is_n_iter_not_reached = loop_par['i'] < solver_settings['max_iter_boundary_method']

        is_max_iter_reached_and_not_finished = jnp.logical_and(
            jnp.logical_not(is_n_iter_not_reached),
            jnp.logical_not(loop_par['is_finished']),
        )

        is_continue_iteration = jnp.logical_and(
            jnp.logical_not(loop_par['is_abort']),
            jnp.logical_and(jnp.logical_not(loop_par['is_finished']), is_n_iter_not_reached)
        )

        if verbose:
            lax.cond(loop_par['is_abort'], lambda: jax.debug.print("-> abort as convergence has stopped"), lambda: None)
            if print_errors:
                lax.cond(
                    is_max_iter_reached_and_not_finished,
                    lambda: jax.debug.print("❌ max. iterations reached without a feasible solution"), lambda: None
                )
                lax.cond(
                    jnp.logical_not(loop_par['is_X_finite']), lambda: jax.debug.print("❌ found non finite numerics"), lambda: None
                )

        return is_continue_iteration

    # loop
    loop_par = {
        'is_finished': jnp.array(False, dtype=jnp.bool_),
        'is_abort': jnp.array(False, dtype=jnp.bool_),
        'is_X_finite': jnp.array(True, dtype=jnp.bool_),
        'variables': variables,
        'parameters_of_dynamic_model': parameters_of_dynamic_model,
        'penality_parameter_trace': penality_parameter_trace,
        'opt_c_eq': opt_c_eq,
        'i': i,
        'verification_state': verification_state_init,
        'tol_inner': tol_inner,
    }

    loop_par = lax.while_loop(loop_cond, loop_body, loop_par)  # loop

    n_iter = loop_par['i']

    return loop_par['variables'], loop_par['opt_c_eq'], n_iter, loop_par['verification_state']


def run_outer_loop_solver(
    variables, parameters_of_dynamic_model, solver_settings, trace_init, objective_, verification_fn_, max_float32_iterations,
    enable_float64, verbose
):
    """
        execute the solution finding process
    """

    opt_c_eq = solver_settings['c_eq_init']
    i = 0
    verification_state = (trace_init, jnp.array(0, dtype=jnp.bool_))

    # iterations that are performed using float32 datatypes
    if max_float32_iterations > 0:
        variables, opt_c_eq, n_iter_f32, verification_state = _run_outer_loop(
            i,
            variables,
            parameters_of_dynamic_model,
            solver_settings['penality_parameter_trace'],
            jnp.array(opt_c_eq, dtype=jnp.float32),
            verification_state,
            solver_settings,
            objective_,
            verification_fn_,
            verbose,
            False,  # show_errors
            target_dtype=jnp.float32
        )

        i = i + n_iter_f32

        if verbose:
            jax.debug.print(
                "👉 switching to higher numerical precision after {n_iter_f32} iterations: float32 --> float64",
                n_iter_f32=n_iter_f32
            )

    # iterations that are performed using float64 datatypes
    if enable_float64:
        variables, opt_c_eq, n_iter_f64, verification_state = _run_outer_loop(
            i,
            variables,
            parameters_of_dynamic_model,
            solver_settings['penality_parameter_trace'],
            jnp.array(opt_c_eq, dtype=jnp.float64),
            verification_state,
            solver_settings,
            objective_,
            verification_fn_,
            verbose,
            True if verbose else False,  # show_errors
            target_dtype=jnp.float64
        )
        i = i + n_iter_f64

    n_iter = i
    variables_star = variables
    trace = get_trace_data(verification_state[0])

    is_converged = verification_state[1]

    return variables_star, is_converged, n_iter, trace
