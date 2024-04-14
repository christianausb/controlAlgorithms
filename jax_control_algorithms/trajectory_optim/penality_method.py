import jax
import jax.numpy as jnp

#from jax_control_algorithms.common import *
from jax_control_algorithms.jax_helper import *

from jax_control_algorithms.trajectory_optim.boundary_function import boundary_fn
from jax_control_algorithms.trajectory_optim.dynamics_constraints import eval_dynamics_equality_constraints
from jax_control_algorithms.trajectory_optim.cost_function import evaluate_cost
"""
    https://en.wikipedia.org/wiki/Penalty_method
"""


def _objective(variables, parameters_passed_to_solver, static_parameters):

    K, parameters, x0, penalty_parameter, opt_c_eq = parameters_passed_to_solver
    f, terminal_constraints, inequality_constraints, cost, running_cost = static_parameters
    X, U = variables

    n_steps = X.shape[0]
    assert U.shape[0] == n_steps

    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = eval_dynamics_equality_constraints(f, terminal_constraints, X, U, K, x0, parameters).reshape(-1)
    c_ineq = inequality_constraints(X, U, K, parameters).reshape(-1)

    # equality constraints using penalty method
    J_equality_costs = opt_c_eq * jnp.mean((c_eq.reshape(-1))**2)

    # eval cost function of problem definition
    J_cost_function = evaluate_cost(f, cost, running_cost, X, U, K, parameters)

    # apply boundary costs (boundary function)
    J_boundary_costs = jnp.mean(boundary_fn(c_ineq, penalty_parameter, 11, True))

    return J_equality_costs + J_cost_function + J_boundary_costs, c_eq


def eval_objective_of_penalty_method(variables, parameters, static_parameters):
    return _objective(variables, parameters, static_parameters)[0]


def eval_feasibility_metric_of_penalty_method(variables, parameters_of_dynamic_model, static_parameters):
    """
        evaluate the correctness of the given solution candidate (variables)

        Check how well
            - the equality, and
            - the inequality
        constraints are fulfilled. For the inequality constraints it is verify if the
        solution candidate is inside the boundaries defined by the constraints.
    """
    K, parameters, x0 = parameters_of_dynamic_model
    f, terminal_constraints, inequality_constraints, cost, running_cost = static_parameters
    X, U = variables

    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = eval_dynamics_equality_constraints(f, terminal_constraints, X, U, K, x0, parameters)
    c_ineq = inequality_constraints(X, U, K, parameters)

    #
    metric_c_eq = jnp.max(jnp.abs(c_eq))

    # check for violations of the boundary
    metric_c_ineq = jnp.max(-jnp.where(c_ineq > 0, 0, c_ineq))

    neq_tol = 0.0001
    is_solution_inside_boundaries = metric_c_ineq < neq_tol  # check if the solution is inside (or close) to the boundary

    return metric_c_eq, is_solution_inside_boundaries


def _check_monotonic_convergence(i, trace):
    """
        Check the monotonic convergence of the error for the equality constraints 
    """
    trace_data = get_trace_data(trace)

    # As being in the 2nd iteration, compare to prev. metric and see if it got smaller
    is_metric_check_active = i > 2

    def true_fn(par):
        i, trace = par

        delta_max_eq_error = trace[0][i] - trace[0][i - 1]
        is_abort = delta_max_eq_error >= 0

        return is_abort

    def false_fn(par):
        return False

    is_not_monotonic = lax.cond(is_metric_check_active, true_fn, false_fn, (i, trace_data))

    return is_not_monotonic


def _eval_eq_constraints_improvement(i, trace):
    """
        return the change of the error for the equality constraints between the iteration i and i-1
    """

    trace_data = get_trace_data(trace)

    def true_fn(par):
        i, trace = par

        normalized_equality_error_before = trace[0][i - 1]
        normalized_equality_error_after = trace[0][i]

        normalized_equality_error_change = normalized_equality_error_before - normalized_equality_error_after
        normalized_equality_error_gain = normalized_equality_error_before / normalized_equality_error_after

        if False:
            jax.debug.print(
                "normalized_equality_error_before={normalized_equality_error_before} normalized_equality_error_after={normalized_equality_error_after}",
                normalized_equality_error_before=normalized_equality_error_before, normalized_equality_error_after=normalized_equality_error_after
            )

        return normalized_equality_error_change, normalized_equality_error_gain

    def false_fn(par):
        return 0.0, 0.0

    return lax.cond(i >= 1, true_fn, false_fn, (i, trace_data))


def verify_convergence_of_iteration(
    verification_state,
    i,
    n_outer_iterations_target,
    res_inner,
    variables,
    parameters_of_dynamic_model,
    penalty_parameter,
    opt_c_eq,  # blub
    feasibility_metric_fn,
    eq_tol,
    verbose: bool
):
    """
        verify the feasibility of the current state of the solution. This function is executed 
        for each iteration of the outer optimization loop.
    """

    trace, _, = verification_state

    #
    is_X_finite = jnp.isfinite(variables[0]).all()
    is_abort_because_of_nonfinite = jnp.logical_not(is_X_finite)

    # verify step
    max_eq_error, is_solution_inside_boundaries = feasibility_metric_fn(variables, parameters_of_dynamic_model)
    n_iter_inner = res_inner.state.iter_num

    #
    normalized_equality_error = max_eq_error / eq_tol

    # verify metrics and check for convergence
    is_equality_constraints_fulfilled = normalized_equality_error < 1.0  # max_eq_error < eq_tol

    is_converged = jnp.logical_and(is_equality_constraints_fulfilled, is_solution_inside_boundaries)

    # trace
    X, U = variables
    trace_next, is_trace_appended = append_to_trace(
        trace, (normalized_equality_error, 1.0 * is_solution_inside_boundaries, n_iter_inner, X, U)
    )
    verification_state_next = (trace_next, is_converged)

    #
    #
    #
    #
    #
    #
    #
    #

    # check for monotonic convergence of the equality constraints
    #
    #
    # NOTE: it is ok if it is not monotonic, the case of step back, the control needs to increase the parameter for the
    # equality constraints
    #
    is_not_monotonic = jnp.logical_and(
        _check_monotonic_convergence(i, trace_next),
        jnp.logical_not(is_converged),
    )

    # measure the improvement of eq-constraints fulfillment
    # ideally, this metric always decreases
    normalized_equality_error_change, normalized_equality_error_gain = _eval_eq_constraints_improvement(i, trace_next)

    # is_abort = jnp.logical_or(is_abort_because_of_nonfinite, is_not_monotonic)
    is_abort = is_abort_because_of_nonfinite

    #
    # control

    def _control_gamma_eq(
        i,  # loop index
        gamma_eq,
        is_equality_constraints_fulfilled,
        normalized_equality_error,
        normalized_equality_error_change,
        normalized_equality_error_gain,
        n_outer_iterations_target
    ):
        # normalized_equality_error --> 1 in n_outer_iterations_target
        #
        # normalized_equality_error / lambda ^ n_outer_iterations_target < 1.0
        # normalized_equality_error = lambda ^ n_outer_iterations_target

        n_iter_left = n_outer_iterations_target - i
        lam = jnp.where(n_iter_left > 3, normalized_equality_error**(1 / n_iter_left), 1.7)

        jax.debug.print(
            "lam={lam} normalized_equality_error={normalized_equality_error} n_outer_iterations_target={n_outer_iterations_target}",
            lam=lam,
            normalized_equality_error=normalized_equality_error,
            n_outer_iterations_target=n_outer_iterations_target
        )

        #        lam = 1.6

        _lam = lam * 1.0

        gamma_eq_next = gamma_eq * jnp.where(is_equality_constraints_fulfilled, 1.0, _lam)

        return gamma_eq_next

    # update opt_c_eq: in case the equality constraints are not satisfies yet, increase opt_c_eq by multiplication with lam > 1
    # otherwise leave opt_c_eq untouched.
    # opt_c_eq_next = loop_var['opt_c_eq'] * jnp.where(is_equality_constraints_fulfilled, 1.0, loop_var['lam'])
    opt_c_eq_next = _control_gamma_eq(
        i, opt_c_eq, is_equality_constraints_fulfilled, normalized_equality_error, normalized_equality_error_change,
        normalized_equality_error_gain, n_outer_iterations_target
    )

    #
    #
    #
    #
    #
    #

    i_best = None

    if verbose:
        jax.debug.print(
            "🔄 it={i} \t (sub iter={n_iter_inner})\tt={penalty_parameter} \teq_error/eq_tol={normalized_equality_error}  gain={normalized_equality_error_gain} change={normalized_equality_error_change} \tinside bounds: {is_solution_inside_boundaries}",
            i=i,
            penalty_parameter=my_to_int(my_round(penalty_parameter, decimals=0)),
            normalized_equality_error=my_to_int(my_round(100 * normalized_equality_error, decimals=0)),
            normalized_equality_error_gain=normalized_equality_error_gain,
            normalized_equality_error_change=normalized_equality_error_change,
            n_iter_inner=n_iter_inner,
            is_solution_inside_boundaries=is_solution_inside_boundaries,
        )

        if False:  # additional info (for debugging purposes)
            jax.debug.print(
                "   is_abort_because_of_nonfinite={is_abort_because_of_nonfinite} is_not_monotonic={is_not_monotonic}) " +
                "is_eq_converged={is_eq_converged}, is_solution_inside_boundaries={is_solution_inside_boundaries}",
                is_abort_because_of_nonfinite=is_abort_because_of_nonfinite,
                is_not_monotonic=is_not_monotonic,
                is_eq_converged=is_equality_constraints_fulfilled,
                is_solution_inside_boundaries=is_solution_inside_boundaries,
            )

    # verification_state, is_finished, is_abort, i_best
    return (
        verification_state_next, is_converged, is_equality_constraints_fulfilled, is_abort, is_X_finite, i_best, max_eq_error,
        normalized_equality_error_change, normalized_equality_error_gain, opt_c_eq_next
    )
