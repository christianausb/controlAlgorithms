import jax
import jax.numpy as jnp
import jaxopt
from functools import partial

from jax_control_algorithms.jax_helper import *
from jax_control_algorithms.trajectory_optim.penality_method import control_convergence_of_iteration
from jax_control_algorithms.trajectory_optim.problem_definition import *
from jax_control_algorithms.trajectory_optim.penality_method import eval_objective_of_penalty_method, eval_objective_of_penalty_method_2
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


def _print_loop_info(loop_par: OuterLoopVariables, is_max_iter_reached_and_not_finished, print_errors):
    lax.cond(loop_par.is_abort, lambda: jax.debug.print("-> abort as convergence has stopped"), lambda: None)
    if print_errors:
        lax.cond(
            is_max_iter_reached_and_not_finished,
            lambda: jax.debug.print("❌ max. iterations reached without a feasible solution"), lambda: None
        )
        lax.cond(jnp.logical_not(loop_par.is_X_finite), lambda: jax.debug.print("❌ found non finite numerics"), lambda: None)


def _iterate(loop_var: OuterLoopVariables, functions, solver_settings : SolverSettings, verbose: bool):

    i = loop_var.i

    # get the penalty parameter
    penalty_parameter = loop_var.penalty_parameter_trace[i]
    n_outer_iterations_target = loop_var.penalty_parameter_trace.shape[0]
    is_maximal_penalty_parameter_reached = i >= n_outer_iterations_target - 1

    #
    parameters_passed_to_inner_solver = loop_var.parameters_of_dynamic_model + (
        penalty_parameter,
        loop_var.opt_c_eq,
    )

    # run inner solver
    # objective_fn = eval_feasibility_metric_of_penalty_method(variables, parameters_of_dynamic_model, functions : Functions)
    objective_fn = partial(
        eval_objective_of_penalty_method,
        functions=functions
    )

    gd = jaxopt.BFGS(fun=objective_fn, value_and_grad=False, tol=loop_var.tol_inner, maxiter=solver_settings.max_iter_inner)
    res = gd.run(loop_var.variables, parameters=parameters_passed_to_inner_solver)
    variables_updated_by_inner_solver = res.params

    # eval objective function on found solution

    (
        J_equality_costs, J_cost_function, J_boundary_costs, 
        c_eq, c_ineq,
    ) = eval_objective_of_penalty_method_2(
        variables=variables_updated_by_inner_solver, parameters=parameters_passed_to_inner_solver, functions=functions
    )

    if False:
        jax.debug.print(
                    "   J_equality_costs={J_equality_costs} J_cost_function={J_cost_function} J_boundary_costs={J_boundary_costs}",
                    J_equality_costs=J_equality_costs,
                    J_cost_function=J_cost_function,
                    J_boundary_costs=J_boundary_costs,
                )
    #

    # run verify the solution and control the convergence of to the equality constraints
    (
        controller_state_next, is_equality_constraints_fulfilled, is_abort, is_X_finite, i_best,
        max_eq_error, opt_c_eq_next, lam
    ) = control_convergence_of_iteration(
        loop_var.controller_state,
        i,
        n_outer_iterations_target,
        res,
        variables_updated_by_inner_solver,
        loop_var.parameters_of_dynamic_model,
        penalty_parameter,
        loop_var.opt_c_eq,
        loop_var.lam,
        functions,
        eq_tol=solver_settings.eq_tol,
        verbose=verbose,
    )

    # update the state of the optimization variables in case the outer loop shall not be aborted
    variables_next = tree_where(is_abort, loop_var.variables, variables_updated_by_inner_solver)

    # solution found?
    is_finished = jnp.logical_and(controller_state_next.is_converged, is_maximal_penalty_parameter_reached)

    return variables_next, controller_state_next, opt_c_eq_next, lam, is_finished, is_abort, is_X_finite


def _run_outer_loop(
    i, variables, model_to_solve: ModelToSolve, opt_c_eq, verification_state_init, solver_settings : SolverSettings, verbose, print_errors,
    target_dtype
):
    """
        Execute the outer loop of the optimization process: herein in each iteration, the parameters of the
        boundary function and the quality cost weight factor are adjusted so that in the final iteration 
        the equality and inequality constraints are fulfilled.
    """

    # convert dtypes to the target datatype used in the computation
    (
        variables,
        # model_to_solve,
        penalty_parameter_trace,
        opt_c_eq,
        verification_state_init,
        tol_inner,
    ) = convert_dtype(
        (
            variables,
            # model_to_solve.parameters_of_dynamic_model, # model_to_solve, # model_to_solve.
            solver_settings.penalty_parameter_trace,
            opt_c_eq,
            verification_state_init,
            solver_settings.tol_inner,
        ),
        target_dtype
    )

    # loop:
    def run_outer_loop_body(loop_var: OuterLoopVariables):

        # loop iteration variable i
        i = loop_var.i

        variables_next, verification_state_next, opt_c_eq_next, lam, is_finished, is_abort, is_X_finite = _iterate(
            loop_var, model_to_solve.functions, solver_settings, verbose
        )

        if verbose:
            lax.cond(is_finished, lambda: jax.debug.print("✅ found feasible solution"), lambda: None)

        loop_var = OuterLoopVariables(
            is_finished=is_finished,
            is_abort=is_abort,
            is_X_finite=is_X_finite,
            variables=variables_next,
            parameters_of_dynamic_model=loop_var.parameters_of_dynamic_model,
            penalty_parameter_trace=penalty_parameter_trace,
            opt_c_eq=opt_c_eq_next,
            lam=lam,
            i=loop_var.i + 1,
            controller_state=verification_state_next,
            tol_inner=loop_var.tol_inner,
        )

        return loop_var

    def eval_outer_loop_condition(loop_var: OuterLoopVariables):
        is_n_iter_not_reached = loop_var.i < solver_settings.max_iter_boundary_method

        is_max_iter_reached_and_not_finished = jnp.logical_and(
            jnp.logical_not(is_n_iter_not_reached),
            jnp.logical_not(loop_var.is_finished),
        )

        is_continue_iteration = jnp.logical_and(
            jnp.logical_not(loop_var.is_abort), jnp.logical_and(jnp.logical_not(loop_var.is_finished), is_n_iter_not_reached)
        )

        if verbose:
            _print_loop_info(loop_var, is_max_iter_reached_and_not_finished, print_errors)

        return is_continue_iteration

    # variables that are passed amount the loop-iterations
    loop_var = OuterLoopVariables(
        is_finished=jnp.array(False, dtype=jnp.bool_),
        is_abort=jnp.array(False, dtype=jnp.bool_),
        is_X_finite=jnp.array(True, dtype=jnp.bool_),
        variables=variables,
        parameters_of_dynamic_model=model_to_solve.parameters_of_dynamic_model,
        penalty_parameter_trace=penalty_parameter_trace,
        opt_c_eq=opt_c_eq,
        lam=jnp.array(jnp.nan),
        i=i,
        controller_state=verification_state_init,
        tol_inner=tol_inner,
    )

    loop_var: OuterLoopVariables = lax.while_loop(eval_outer_loop_condition, run_outer_loop_body, loop_var)

    n_iter = loop_var.i

    return loop_var.variables, loop_var.opt_c_eq, n_iter, loop_var.controller_state


def run_outer_loop_solver(
    variables, model_to_solve, solver_settings : SolverSettings, trace_init, max_float32_iterations, enable_float64, verbose
):
    """
        execute the solution finding process
    """

    opt_c_eq = solver_settings.c_eq_init
    i = 0
    verification_state = ConvergenceControllerState(trace=trace_init, is_converged=jnp.array(0, dtype=jnp.bool_))

    # iterations that are performed using float32 datatypes
    if max_float32_iterations > 0:
        variables, opt_c_eq, n_iter_f32, verification_state = _run_outer_loop(
            i,
            variables,
            model_to_solve,
            jnp.array(opt_c_eq, dtype=jnp.float32),
            verification_state,
            solver_settings,
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
            model_to_solve,
            jnp.array(opt_c_eq, dtype=jnp.float64),
            verification_state,
            solver_settings,
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
