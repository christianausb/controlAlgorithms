import jax
import jax.numpy as jnp

#from jax_control_algorithms.common import *
from jax_control_algorithms.jax_helper import *

from jax_control_algorithms.trajectory_optim.boundary_function import boundary_fn
from jax_control_algorithms.trajectory_optim.dynamics_constraints import eval_dynamics_equality_constraints
from jax_control_algorithms.trajectory_optim.cost_function import evaluate_cost
from jax_control_algorithms.trajectory_optim.problem_definition import Functions, ConvergenceControllerState
"""
    https://en.wikipedia.org/wiki/Penalty_method
"""


def _objective(variables, parameters_passed_to_solver, functions : Functions):
    """
        compute the objective function including
        
        - a penalty for equality constraints
        - a penalty for boundaries
        - the cost function of the problem to solve
    """

    K, parameters, x0, penalty_parameter, opt_c_eq = parameters_passed_to_solver
    X, U = variables

    n_steps = X.shape[0]
    assert U.shape[0] == n_steps

    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = eval_dynamics_equality_constraints(functions.f, functions.terminal_constraints, X, U, K, x0, parameters).reshape(-1)
    c_ineq = functions.inequality_constraints(X, U, K, parameters).reshape(-1)

    # equality constraints using penalty method
    J_equality_costs = opt_c_eq * jnp.mean((c_eq.reshape(-1))**2)

    # eval cost function of problem definition
    J_cost_function = evaluate_cost(functions.f, functions.cost, functions.running_cost, X, U, K, parameters)

    # apply boundary costs (boundary function)
    J_boundary_costs = jnp.mean(boundary_fn(c_ineq, penalty_parameter, 11, True))

    return (
        J_equality_costs, J_cost_function, J_boundary_costs, 
        c_eq, c_ineq,
    )


def eval_objective_of_penalty_method(variables, parameters, functions : Functions):
    J_equality_costs, J_cost_function, J_boundary_costs, _, _ = _objective(variables, parameters, functions)

    return J_equality_costs + J_cost_function + J_boundary_costs

    #return _objective(variables, parameters, functions)[0]

def eval_objective_of_penalty_method_2(variables, parameters, functions : Functions):
    return _objective(variables, parameters, functions)


def eval_feasibility_metric_of_penalty_method(variables, parameters_of_dynamic_model, functions : Functions):
    """
        evaluate the correctness of the given solution candidate (variables)

        Check how well
            - the equality, and
            - the inequality
        constraints are fulfilled. For the inequality constraints it is verify if the
        solution candidate is inside the boundaries defined by the constraints.
    """
    K, parameters, x0 = parameters_of_dynamic_model
    X, U = variables

    # get equality constraint. The constraints are fulfilled of all elements of c_eq are zero
    c_eq = eval_dynamics_equality_constraints( functions.f, functions.terminal_constraints, X, U, K, x0, parameters)
    c_ineq = functions.inequality_constraints(X, U, K, parameters)

    #
    metric_c_eq = jnp.max(jnp.abs(c_eq))

    # check for violations of the boundary
    metric_c_ineq = jnp.max(-jnp.where(c_ineq > 0, 0, c_ineq))

    neq_tol = 0.0001
    is_solution_inside_boundaries = metric_c_ineq < neq_tol  # check if the solution is inside (or close) to the boundary

    return metric_c_eq, is_solution_inside_boundaries


def _eval_eq_constraints_improvement(i, trace):
    """
        return the change of the error for the equality constraints between the iteration i and i-1
    """

    trace_data = get_trace_data(trace)

    def true_fn(par):
        i, trace = par

        normalized_equality_error_before = trace[0][i - 1]
        normalized_equality_error_after = trace[0][i]

        normalized_equality_error_gain = normalized_equality_error_before / normalized_equality_error_after

        return normalized_equality_error_gain

    def false_fn(par):
        return 0.0

    return lax.cond(i >= 1, true_fn, false_fn, (i, trace_data))

def _control_gamma_eq(
    i,  # loop index
    gamma_eq,
    lam_prev,
    is_equality_constraints_fulfilled,
    normalized_equality_error,
    normalized_equality_error_gain,
    n_outer_iterations_target
):
    # normalized_equality_error --> 1 in n_outer_iterations_target
    #
    # normalized_equality_error / lambda ^ n_outer_iterations_target < 1.0
    # normalized_equality_error = lambda ^ n_outer_iterations_target

    n_iter_left = n_outer_iterations_target - i

    lam = jnp.where(n_iter_left > 3, normalized_equality_error**(1 / n_iter_left), lam_prev)

    if False:
        jax.debug.print(
            "lam={lam} lam_prev={lam_prev} normalized_equality_error={normalized_equality_error} n_outer_iterations_target={n_outer_iterations_target}",
            lam=lam,
            lam_prev=lam_prev,
            normalized_equality_error=normalized_equality_error,
            n_outer_iterations_target=n_outer_iterations_target
        )

    gamma_eq_next = gamma_eq * jnp.where(is_equality_constraints_fulfilled, 1.0, lam)

    # NOTE: normalized_equality_error_gain shall be close to lam_prev
    # This could be used for monitoring the convergence. 

    return gamma_eq_next, lam

def control_convergence_of_iteration(
    controller_state : ConvergenceControllerState,
    i,
    n_outer_iterations_target,
    res_inner,
    variables,
    parameters_of_dynamic_model,
    penalty_parameter,
    opt_c_eq,
    lam,
    functions,
    eq_tol,
    verbose: bool
):
    """
        verify the feasibility of the current state of the solution. This function is executed 
        for each iteration of the outer optimization loop.
    """

    trace = controller_state.trace

    #
    is_X_finite = jnp.isfinite(variables[0]).all()
    is_abort_because_of_nonfinite = jnp.logical_not(is_X_finite)

    # verify step
    max_eq_error, is_solution_inside_boundaries = eval_feasibility_metric_of_penalty_method(variables, parameters_of_dynamic_model, functions)
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
    verification_state_next = ConvergenceControllerState(trace=trace_next, is_converged=is_converged)

    # measure the improvement of eq-constraints fulfillment
    # ideally, this metric always decreases
    normalized_equality_error_gain = _eval_eq_constraints_improvement(i, trace_next)

    # is_abort = jnp.logical_or(is_abort_because_of_nonfinite, is_not_monotonic)
    is_abort = is_abort_because_of_nonfinite

    #
    # control
    #

    opt_c_eq_next, lam_next = _control_gamma_eq(
        i, opt_c_eq, lam, is_equality_constraints_fulfilled, normalized_equality_error,
        normalized_equality_error_gain, n_outer_iterations_target
    )

    i_best = None

    if verbose:
        jax.debug.print(
            "🔄 it={i} \t (sub iter={n_iter_inner})\tt={penalty_parameter} \teq_error/eq_tol={normalized_equality_error}  gain={normalized_equality_error_gain} lambda={lam} \tinside bounds: {is_solution_inside_boundaries}",
            i=i,
            penalty_parameter=my_to_int(my_round(penalty_parameter, decimals=0)),
            normalized_equality_error=my_to_int(my_round(100 * normalized_equality_error, decimals=0)),
            normalized_equality_error_gain=normalized_equality_error_gain,
            lam=lam_next,
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

    return (
        verification_state_next, is_equality_constraints_fulfilled, is_abort, is_X_finite, i_best, max_eq_error,
        opt_c_eq_next, lam_next
    )
