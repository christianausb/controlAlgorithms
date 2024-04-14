import jax
import jax.numpy as jnp
import jaxopt

from jax_control_algorithms.jax_helper import *
"""
    Implements the nested solver loops

    - An outer loop varies parameters for the penalty method, while
    - the inner solver (BGFS) is executed in each iteration of the outer loop to solve 
      the non-linear optimization problem that is defined by the penalty method


    Task
    ----

    Minimize the cost function J(x) under the constraints

        h(x) = 0
        g(x) < 0

    The problem is transformed into a non-linear optimization problem without constraints:

        min_x  J(x)  +  gamma_eq * mean( h_i(x) ^ 2 )  +  gamma_neq * mean( s(g_i(x)) )

    Herein, s is the boundary function that approximates an ideal boundary

        s*(x < 0) = infinity
        s*(x >= 0) = 0

"""


def _print_loop_info(loop_par, is_max_iter_reached_and_not_finished, print_errors):
    lax.cond(loop_par['is_abort'], lambda: jax.debug.print("-> abort as convergence has stopped"), lambda: None)
    if print_errors:
        lax.cond(
            is_max_iter_reached_and_not_finished,
            lambda: jax.debug.print("❌ max. iterations reached without a feasible solution"), lambda: None
        )
        lax.cond(jnp.logical_not(loop_par['is_X_finite']), lambda: jax.debug.print("❌ found non finite numerics"), lambda: None)


def _iterate(i, loop_var, solver_settings, objective_fn, verification_fn):

    # get the penalty parameter
    penalty_parameter = loop_var['penalty_parameter_trace'][i]
    n_outer_iterations_target = loop_var['penalty_parameter_trace'].shape[0]
    is_maximal_penalty_parameter_reached = i >= n_outer_iterations_target - 1

    #
    parameters_passed_to_inner_solver = loop_var['parameters_of_dynamic_model'] + (
        penalty_parameter,
        loop_var['opt_c_eq'],
    )

    # run inner solver
    gd = jaxopt.BFGS(fun=objective_fn, value_and_grad=False, tol=loop_var['tol_inner'], maxiter=solver_settings['max_iter_inner'])
    res = gd.run(loop_var['variables'], parameters=parameters_passed_to_inner_solver)
    variables_updated_by_inner_solver = res.params

    # run callback to verify the solution
    (
        verification_state_next, is_solution_feasible, is_equality_constraints_fulfilled, is_abort, is_X_finite, i_best,
        max_eq_error, normalized_equality_error_change, normalized_equality_error_gain, opt_c_eq_next
    ) = verification_fn(
        loop_var['verification_state'], i, n_outer_iterations_target, res, variables_updated_by_inner_solver, loop_var['parameters_of_dynamic_model'],
        penalty_parameter, loop_var['opt_c_eq']
    )

    # update the state of the optimization variables in case the outer loop shall not be aborted
    variables_next = tree_where(is_abort, loop_var['variables'], variables_updated_by_inner_solver)

    #
    #
    #
    #

    # trace, _ = verification_state_next

    # def _control_gamma_eq(
    #     gamma_eq, is_equality_constraints_fulfilled, lam, max_eq_error, normalized_equality_error_change,
    #     normalized_equality_error_gain
    # ):

    #     _lam = lam * 1.0

    #     gamma_eq_next = gamma_eq * jnp.where(is_equality_constraints_fulfilled, 1.0, _lam)

    #     return gamma_eq_next

    # # update opt_c_eq: in case the equality constraints are not satisfies yet, increase opt_c_eq by multiplication with lam > 1
    # # otherwise leave opt_c_eq untouched.
    # # opt_c_eq_next = loop_var['opt_c_eq'] * jnp.where(is_equality_constraints_fulfilled, 1.0, loop_var['lam'])
    # opt_c_eq_next = _control_gamma_eq(
    #     loop_var['opt_c_eq'], is_equality_constraints_fulfilled, loop_var['lam'], max_eq_error, normalized_equality_error_change,
    #     normalized_equality_error_gain
    # )

    #
    #
    #
    #

    # solution found?
    is_finished = jnp.logical_and(is_solution_feasible, is_maximal_penalty_parameter_reached)

    return variables_next, verification_state_next, opt_c_eq_next, is_finished, is_abort, is_X_finite


def _run_outer_loop(
    i, variables, parameters_of_dynamic_model, opt_c_eq, verification_state_init, solver_settings, objective_fn, verification_fn,
    verbose, print_errors, target_dtype
):
    """
        Execute the outer loop of the optimization process: herein in each iteration, the parameters of the
        boundary function and the quality cost weight factor are adjusted so that in the final iteration 
        the equality and inequality constraints are fulfilled.
    """

    # convert dtypes to the target datatype used in the computation
    (
        variables,
        parameters_of_dynamic_model,
        penalty_parameter_trace,
        opt_c_eq,
        verification_state_init,
        lam,
        tol_inner,
    ) = convert_dtype(
        (
            variables,
            parameters_of_dynamic_model,
            solver_settings['penalty_parameter_trace'],
            opt_c_eq,
            verification_state_init,
            solver_settings['lam'],
            solver_settings['tol_inner'],
        ), target_dtype
    )

    # loop:
    def run_outer_loop_body(loop_var):

        # loop iteration variable i
        i = loop_var['i']

        variables_next, verification_state_next, opt_c_eq_next, is_finished, is_abort, is_X_finite = _iterate(
            i, loop_var, solver_settings, objective_fn, verification_fn
        )

        if verbose:
            lax.cond(is_finished, lambda: jax.debug.print("✅ found feasible solution"), lambda: None)

        loop_var = {
            'is_finished': is_finished,
            'is_abort': is_abort,
            'is_X_finite': is_X_finite,
            'variables': variables_next,
            'parameters_of_dynamic_model': loop_var['parameters_of_dynamic_model'],
            'penalty_parameter_trace': penalty_parameter_trace,
            'opt_c_eq': opt_c_eq_next,
            'i': loop_var['i'] + 1,
            'verification_state': verification_state_next,
            'tol_inner': loop_var['tol_inner'],
            'lam': loop_var['lam'],
        }

        return loop_var

    def eval_outer_loop_condition(loop_var):
        is_n_iter_not_reached = loop_var['i'] < solver_settings['max_iter_boundary_method']

        is_max_iter_reached_and_not_finished = jnp.logical_and(
            jnp.logical_not(is_n_iter_not_reached),
            jnp.logical_not(loop_var['is_finished']),
        )

        is_continue_iteration = jnp.logical_and(
            jnp.logical_not(loop_var['is_abort']),
            jnp.logical_and(jnp.logical_not(loop_var['is_finished']), is_n_iter_not_reached)
        )

        if verbose:
            _print_loop_info(loop_var, is_max_iter_reached_and_not_finished, print_errors)

        return is_continue_iteration

    # variables that are passed amount the loop-iterations
    loop_var = {
        'is_finished': jnp.array(False, dtype=jnp.bool_),
        'is_abort': jnp.array(False, dtype=jnp.bool_),
        'is_X_finite': jnp.array(True, dtype=jnp.bool_),
        'variables': variables,
        'parameters_of_dynamic_model': parameters_of_dynamic_model,
        'penalty_parameter_trace': penalty_parameter_trace,
        'opt_c_eq': opt_c_eq,
        'i': i,
        'verification_state': verification_state_init,
        'tol_inner': tol_inner,
        'lam': lam,
    }

    loop_var = lax.while_loop(eval_outer_loop_condition, run_outer_loop_body, loop_var)

    n_iter = loop_var['i']

    return loop_var['variables'], loop_var['opt_c_eq'], n_iter, loop_var['verification_state']


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
            jnp.array(opt_c_eq, dtype=jnp.float32),
            verification_state,
            solver_settings,
            objective_,
            verification_fn_,
            verbose,
            True if verbose else False,  # show_errors
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
