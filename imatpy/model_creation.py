"""
Submodule with functions for creating a context specific model using iMAT
"""

# Potential methods:
# - imat_restrictions
# - simple_bounds
# - eliminate_below_threshold
# - fva
# - MILP

# Standard Library Imports
from typing import Union

# External Imports
import cobra
import numpy as np
import pandas as pd
import sympy as sym

# Local Imports
from imatpy.imat import imat, add_imat_constraints, add_imat_objective_

# define defaults for the iMAT functions
DEFAULTS = {
    "epsilon": 1e-2,
    "objective_tolerance": 5e-2,
    "threshold": 1e-3,
    "tolerance": 1e-7,
}


# region: Main Model Creation Function
def generate_model(model: cobra.Model,
                   rxn_weights: Union[pd.Series, dict],
                   method: str = "imat_restrictions",
                   epsilon: float = DEFAULTS["epsilon"],
                   threshold: float = DEFAULTS["threshold"],
                   objective_tolerance: float = DEFAULTS["objective_tolerance"]):
    """
    Generate a context specific model using iMAT.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param method: The method to use for generating the context specific model. Valid methods are:
        'imat_restrictions', 'simple_bounds', 'eliminate_below_threshold', 'fva', 'milp'.
    :type method: str
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :param objective_tolerance: The tolerance for the objective value (used for imat_restrictions and fva methods).
        The objective will be restricted to be within objective_tolerance*objective_value of the optimal objective
        value. (default: 5e-2)
    :type objective_tolerance: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. seealso::
        | :func:`_imat_restrictions` for more information on the imat_restrictions method.
        | :func:`_simple_bounds` for more information on the simple_bounds method.
        | :func:`_eliminate_below_threshold` for more information on the eliminate_below_threshold method.
        | :func:`_fva` for more information on the fva method.
        | :func:`_milp` for more information on the milp method.
    """
    method = _parse_method(method)
    if method == "imat_restrictions":
        return _imat_restrictions(model, rxn_weights, epsilon, threshold, objective_tolerance)
    elif method == "simple_bounds":
        return _simple_bounds(model, rxn_weights, epsilon, threshold)
    elif method == "eliminate_below_threshold":
        return _eliminate_below_threshold(model, rxn_weights, epsilon, threshold)
    elif method == "fva":
        return _fva(model, rxn_weights, epsilon, threshold, objective_tolerance)
    elif method == "milp":
        return _milp(model, rxn_weights, epsilon, threshold)
    else:
        raise ValueError(f"Invalid method: {method}. Valid methods are: 'simple_bounds', 'imat_restrictions', "
                         f"'eliminate_below_threshold', 'fva', 'milp'.")


# endregion: Main Model Creation Function


# region: Model Creation methods
def _imat_restrictions(model, rxn_weights, epsilon, threshold, objective_tolerance):
    """
    Generate a context specific model by adding iMAT constraints, and ensuring iMAT objective value is near optimal.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :param objective_tolerance: The tolerance for the objective value. The objective will be restricted to be within
        objective_tolerance*objective_value of the optimal objective value. (default: 5e-2)
    :type objective_tolerance: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This function first solves the iMAT problem, then adds a constraint to ensure that the iMAT
        objective value is within objective_tolerance*objective_value of the optimal objective value. This model will
        include integer constraints, and so can not be used for sampling. If you want to use the model for sampling,
        use any of the other methods.

    """
    original_objective = model.objective
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    imat_solution = imat_model.optimize()
    imat_obj_val = imat_solution.objective_value
    rh = rxn_weights[rxn_weights > 0].index.tolist()
    rl = rxn_weights[rxn_weights < 0].index.tolist()
    rh_obj = []
    rl_obj = []
    for rxn in rh:  # For each highly expressed reaction
        # Get the forward and reverse variables from the model
        forward_variable = imat_model.solver.variables[f"y_pos_{rxn}"]
        reverse_variable = imat_model.solver.variables[f"y_neg_{rxn}"]
        # Adds the two variables to the rh list which will be used for sum
        rh_obj += [forward_variable, reverse_variable]
    for rxn in rl:  # For each lowly expressed reaction
        variable = imat_model.solver.variables[f"y_pos{rxn}"]
        rl_obj += [variable]  # Note: Only one variable for lowly expressed reactions
    imat_obj_constraint = imat_model.solver.interface.Constraint(
        sym.Add(*rh_obj) + sym.Add(*rl_obj), lb=imat_obj_val - objective_tolerance * imat_obj_val)
    imat_model.solver.add(imat_obj_constraint)
    imat_model.objective = original_objective
    return imat_model


def _simple_bounds(model, rxn_weights, epsilon, threshold):
    """
    Generate a context specific model by setting bounds on reactions based on iMAT solution.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This method first solves the iMAT solution, then for reactions found to be lowly expressed (weight<0), and
        inactive (flux<threshold), the reaction bounds are set to (-threshold, threshold), (0, threshold), or
        (-threshold, 0) depending on reversibility. For reactions found to be highly expressed and active in the
        forward direction (weight>0, flux>epsilon), the reaction bounds are set to (epsilon, ub), or
        (lb, ub) if lb>epsilon. For reactions found to be highly expressed and active in the reverse direction
        (weight>0, flux<-epsilon), the reaction bounds are set to (lb, -epsilon), or (lb, ub) if ub<-epsilon.
        This model will not include integer constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    imat_solution = imat(model, rxn_weights, epsilon, threshold)
    fluxes = imat_solution.fluxes
    rl = rxn_weights[rxn_weights < 0].index.tolist()
    rh = rxn_weights[rxn_weights > 0].index.tolist()
    inactive_reactions = fluxes[(fluxes.abs() <= threshold) & (fluxes.index.isin(rl))]
    forward_active_reactions = fluxes[(fluxes >= epsilon) & (fluxes.index.isin(rh))]
    reverse_active_reactions = fluxes[(fluxes <= -epsilon) & (fluxes.index.isin(rh))]
    for rxn in inactive_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _low_expr_inactive_bounds(reaction.lower_bound, reaction.upper_bound, threshold)
    for rxn in forward_active_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _high_expr_active_bounds(reaction.lower_bound, reaction.upper_bound, epsilon, forward=True)
    for rxn in reverse_active_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _high_expr_active_bounds(reaction.lower_bound, reaction.upper_bound, epsilon, forward=False)
    return updated_model


def _eliminate_below_threshold(model, rxn_weights, epsilon, threshold):
    """
    Generate a context specific model by knocking out reactions found to be inactive by iMAT.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This method first solves the iMAT solution, then for reactions found to be lowly expressed (weight<0), and
        inactive (flux<threshold), the reaction bounds are set to (0, 0). This model will not include integer
        constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    imat_solution = imat(model, rxn_weights, epsilon, threshold)
    fluxes = imat_solution.fluxes
    rl = rxn_weights[rxn_weights < 0].index.tolist()
    inactive_reactions = fluxes[(fluxes.abs() <= threshold) & (fluxes.index.isin(rl))]
    for rxn in inactive_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = (0, 0)  # KO the reaction
    return updated_model


def _fva(model, rxn_weights, epsilon, threshold, objective_tolerance):
    """
    Generate a context specific model by setting bounds on reactions based on FVA for an iMAT model.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :param objective_tolerance: The tolerance for the objective value. The objective will be restricted to be within
        objective_tolerance*objective_value of the optimal objective value. (default: 5e-2)
    :type objective_tolerance: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This method first creates a model with the iMAT constraints, and objective and then performs FVA to find
        the minimum and maximum flux for each reaction which allow for the objective to be within tolerance of optimal.
        These values are then set as the reaction bounds. This model is not guaranteed to have fluxes consistent
        with the optimal iMAT objective. This model will not include integer constraints, and so can be used for
        sampling.
    """
    updated_model = model.copy()
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    fva_res = cobra.flux_analysis.flux_variability_analysis(imat_model, fraction_of_optimum=1 - objective_tolerance)
    reactions = rxn_weights[~np.isclose(rxn_weights, 0)].index.tolist()
    for rxn in reactions:
        reaction = updated_model.reactions.get_by_id(rxn)
        if fva_res.loc[rxn, "status"] == "optimal":
            reaction.bounds = (fva_res.loc[rxn, "minimum"], fva_res.loc[rxn, "maximum"])
    return updated_model


def _milp(model, rxn_weights, epsilon, threshold):
    """
    Generate a context specific model by setting bounds on reactions based on a set of mixed integer linear programming
    problems.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This method first creates a model with the iMAT constraints, and objective and then solves a set of mixed
        integer linear programming problems, where each reaction is set to be inactive, active in the forward direction,
        and active in the reverse direction. The reaction bounds are then set based on the results of the MILP
        problems. This model is not guaranteed to have fluxes consistent with the optimal iMAT objective. This model
        will include integer constraints, and so can not be used for sampling.
    """
    updated_model = model.copy()
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    milp_results = pd.DataFrame(np.nan, columns=["inactive", "forward", "reverse"],
                                index=imat_model.reactions.list_attr("id"), dtype=float)
    for rxn in milp_results.index:
        with imat_model as ko_model:  # Knock out the reaction
            reaction = ko_model.reactions.get_by_id(rxn)
            reaction.bounds = _low_expr_inactive_bounds(reaction.lb, reaction.ub, threshold)
            ko_solution = ko_model.slim_optimize(error_value=np.nan)
        with imat_model as forward_model:
            reaction = forward_model.reactions.get_by_id(rxn)
            reaction.bounds = _milp_force_forward(reaction.lb, reaction.ub, epsilon)
            forward_solution = forward_model.slim_optimize(error_value=np.nan)
        with imat_model as reverse_model:
            reaction = reverse_model.reactions.get_by_id(rxn)
            reaction.bounds = _milp_force_reverse(reaction.lb, reaction.ub, epsilon)
            reverse_solution = reverse_model.slim_optimize(error_value=np.nan)
        milp_results.loc[rxn, :] = [ko_solution, forward_solution,
                                    reverse_solution]
    milp_results["results"] = milp_results.apply(_milp_eval, axis=1).replace(
        {2: -1}).dropna()  # Now 0 is inactive, 1 is forward, -1 is reverse, nan is under determined
    for rxn in milp_results.index:
        if pd.isna(milp_results.loc[rxn, "results"]):  # skip under-determined reactions
            continue  # should never actually happen due to drop na, but here for safety
        reaction = updated_model.reactions.get_by_id(rxn)
        if milp_results.loc[rxn, "results"] == 0:  # inactive
            reaction.bounds = _low_expr_inactive_bounds(reaction.lb, reaction.ub, threshold)
        elif milp_results.loc[rxn, "results"] == 1:  # forward
            reaction.bounds = _milp_force_forward(reaction.lb, reaction.ub, epsilon)
        elif milp_results.loc[rxn, "results"] == -1:  # reverse
            reaction.bounds = _milp_force_reverse(reaction.lb, reaction.ub, epsilon)
    return updated_model


# endregion: Model Creation methods

# region Helper Functions
def _parse_method(method: str) -> str:
    """
    Parse the method string to a valid method name.

    :param method: The method to parse.
    :type method: str
    :return: The parsed method name.
    :rtype: str
    """
    if method.lower() in ["simple", "simple_bounds", "simple bounds", "simple-bounds", "sb"]:
        return "simple_bounds"
    elif method.lower() in ["imat", "imat_restrictions", "imat restrictions", "imat-restrictions", "ir"]:
        return "imat_restrictions"
    elif method.lower() in ["eliminate", "eliminate_below_threshold", "eliminate below threshold",
                            "eliminate-below-threshold", "ebt"]:
        return "eliminate_below_threshold"
    elif method.lower() in ["fva", "flux_variability_analysis", "flux variability analysis",
                            "flux-variability-analysis", "f"]:
        return "fva"
    elif method.lower() in ["milp", "mixed_integer_linear_programming", "mixed integer linear programming",
                            "mixed-integer-linear-programming", "m"]:
        return "milp"
    else:
        raise ValueError(f"Invalid method: {method}. Valid methods are: 'simple_bounds', 'imat_restrictions', "
                         f"'eliminate_below_threshold', 'fva', 'milp'.")


def _low_expr_inactive_bounds(lb: float, ub: float, threshold: float) -> tuple[float, float]:
    """
    Find the new bounds for the reaction if it is inactive.
    """
    new_lb = max(lb, -threshold)
    new_ub = min(ub, threshold)
    return new_lb, new_ub


def _high_expr_active_bounds(lb: float, ub: float, epsilon: float, forward: bool) -> tuple[float, float]:
    """
    Find the new bounds for the reaction if it is active.
    """
    if forward:
        new_lb = max(lb, epsilon)
        new_ub = ub
    else:
        new_lb = lb
        new_ub = min(ub, -epsilon)
    return new_lb, new_ub


def _milp_force_forward(lb: float, ub: float, epsilon: float) -> tuple[float, float]:
    """
    Find the bounds for the reaction if it is forced to be active in the forward direction, when not
    guaranteed to be able to be active in the forward direction.
    """
    if ub <= 0.:
        return 0., 0.
    if ub >= epsilon:
        return max(epsilon, lb), ub
    return 0, ub


def _milp_force_reverse(lb: float, ub: float, epsilon: float) -> tuple[float, float]:
    """
    Find the bounds for the reaction if it is forced to be active in the reverse direction, when not
    guaranteed to be able to be active in the reverse direction.
    """
    if lb >= 0.:
        return 0., 0.
    if lb <= -epsilon:
        return lb, min(-epsilon, ub)
    return lb, 0.


def _milp_eval(milp_res: pd.Series) -> float:
    """
    Function for evaluating the results of the MILP method, to determine if a reaction should be considered active or
    inactive.
    """
    if pd.isna(milp_res).any():
        return np.nan
    if len(np.unique(milp_res)) == 3:  # All three values are unique, return index of greatest value
        np.argmax(milp_res)
    elif (milp_res["forward"] == milp_res["reverse"]) and (milp_res["inactive"] > milp_res["forward"]):
        return 0  # Forced forward, and reverse are the same, and inactive is greater than both, so inactive
    elif (milp_res["inactive"] == milp_res["reverse"]) and (milp_res["forward"] > milp_res["inactive"]):
        return 1  # Forced reverse, and inactive are the same, and forward is greater than both, so forward
    elif (milp_res["inactive"] == milp_res["forward"]) and (milp_res["reverse"] > milp_res["inactive"]):
        return 2  # Forced inactive, and forward are the same, and reverse is greater than both, so reverse
    else:
        return np.nan  # Under-determined case, return nan

# endregion Helper Functions
