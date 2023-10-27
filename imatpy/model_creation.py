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
        forward_variable = model.solver.variables[f"y_pos_{rxn}"]
        reverse_variable = model.solver.variables[f"y_neg_{rxn}"]
        # Adds the two variables to the rh list which will be used for sum
        rh_obj += [forward_variable, reverse_variable]
    for rxn in rl:  # For each lowly expressed reaction
        variable = model.solver.variables[f"y_pos{rxn}"]
        rl_obj += [variable]  # Note: Only one variable for lowly expressed reactions
    imat_obj_constraint = model.solver.interface.Constraint(
        sym.Add(*rh_obj) + sym.Add(*rl_obj), lb=imat_obj_val - objective_tolerance * imat_obj_val)
    imat_model.solver.add(imat_obj_constraint)
    imat_model.objective = original_objective
    return imat_model


def _simple_bounds(model, rxn_weights, epsilon, threshold):
    imat_solution = imat(model, rxn_weights, epsilon, threshold)
    fluxes = imat_solution.fluxes
    rl = rxn_weights[rxn_weights < 0].index.tolist()
    rh = rxn_weights[rxn_weights > 0].index.tolist()
    inactive_reactions = fluxes[(fluxes.abs() <= threshold) & (fluxes.index.isin(rl))]
    forward_active_reactions = fluxes[(fluxes >= epsilon) & (fluxes.index.isin(rh))]
    reverse_active_reactions = fluxes[(fluxes <= -epsilon) & (fluxes.index.isin(rh))]
    for rxn in inactive_reactions.index:
        reaction = model.reactions.get_by_id(rxn)
        lb = reaction.lower_bound
        ub = reaction.upper_bound
        # TODO: Fix bounds


def _eliminate_below_threshold(model, rxn_weights, epsilon, threshold):
    pass


def _fva(model, rxn_weights, epsilon, threshold, objective_tolerance):
    pass


def _milp(model, rxn_weights, epsilon, threshold):
    pass


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
# endregion Helper Functions
