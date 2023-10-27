"""
Submodule with functions for adding iMAT constraints and objectives to a cobra model, and running iMAT
"""
# Standard Library Imports
from typing import Union

# External Imports
import cobra
import numpy as np
import pandas as pd
import sympy as sym

# Local Imports

# define defaults for the iMAT functions
DEFAULTS = {
    "epsilon": 1e-2,
    "threshold": 1e-3,
    "tolerance": 1e-7,
}


# region: Main iMat Function
def imat(model: cobra.Model,
         rxn_weights: Union[pd.Series, dict],
         epsilon: float = DEFAULTS["epsilon"],
         threshold: float = DEFAULTS["threshold"]) -> cobra.Solution:
    """
    Function for performing iMAT analysis. Returns a cobra Solution object, with objective value and fluxes.

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
    :return: A cobra Solution object with the objective value and fluxes.
    :rtype: cobra.Solution
    """
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    return imat_model.optimize()


# endregion: Main iMat Function

# region: iMAT Helper Functions
def add_imat_constraints_(model: cobra.Model,
                          rxn_weights: Union[pd.Series, dict],
                          epsilon: float = DEFAULTS["epsilon"],
                          threshold: float = DEFAULTS["threshold"]) -> cobra.Model:
    """
    Add the IMAT constraints to the model (updates the model in place).

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :return: The updated model.
    :rtype: cobra.Model
    """
    for rxn, weight in rxn_weights.items():
        if np.isclose(weight, 0):  # Don't add any restrictions for 0 weight reactions
            continue
        if weight > 0:  # Add highly expressed constraint
            _imat_pos_weight_(model=model, rxn=rxn, epsilon=epsilon)
        elif weight < 0:  # Add lowly expressed constraint
            _imat_neg_weight_(model=model, rxn=rxn, threshold=threshold)
    return model


def add_imat_constraints(model, rxn_weights, epsilon: float = 1e-3, threshold: float = 1e-4) -> cobra.Model:
    """
    Add the IMAT constraints to the model (returns new model, doesn't update model in place).

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :return: The updated model.
    :rtype: cobra.Model
    """
    imat_model = model.copy()
    add_imat_constraints_(imat_model, rxn_weights, epsilon, threshold)
    return imat_model


def add_imat_objective_(model: cobra.Model, rxn_weights: Union[pd.Series, dict]) -> None:
    """
    Add the IMAT objective to the model (updates the model in place). Model must already have iMAT constraints added.

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :return: None
    """
    if isinstance(rxn_weights, dict):
        rxn_weights = pd.Series(rxn_weights)
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
    imat_obj = model.solver.interface.Objective(sym.Add(*rh_obj) + sym.Add(*rl_obj), direction="max")
    model.objective = imat_obj


def add_imat_objective(model: cobra.Model, rxn_weights: Union[pd.Series, dict]) -> cobra.Model:
    """
    Add the IMAT objective to the model (doesn't change passed model). Model must already have iMAT constraints added.

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :return: None
    """
    imat_model = model.copy()
    add_imat_objective_(imat_model, rxn_weights)
    return imat_model


# endregion: iMAT Helper Functions

# region: Internal Methods
def _imat_pos_weight_(model: cobra.Model, rxn: str, epsilon: float) -> None:
    """
    Internal method for adding positive weight constraints to the model.

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn: The reaction ID to add the constraint to.
    :type rxn: str
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3). Represents the minimum flux for a reaction to
        be considered on.
    :type epsilon: float
    :return: None
    """
    reaction = model.reactions.get_by_id(rxn)
    lb = reaction.lower_bound
    ub = reaction.upper_bound
    reaction_flux = reaction.forward_variable - reaction.reverse_variable
    y_pos = model.solver.interface.Variable(f"y_pos_{reaction.id}", type="binary")
    model.solver.add(y_pos)
    forward_constraint = model.solver.interface.Constraint(reaction_flux + y_pos * (lb - epsilon), lb=lb,
                                                           name=f"forward_constraint_{reaction.id}")
    model.solver.add(forward_constraint)
    y_neg = model.solver.interface.Variable(f"y_neg_{reaction.id}", type="binary")
    model.solver.add(y_neg)
    reverse_constraint = model.solver.interface.Constraint(reaction_flux + y_neg * (ub + epsilon), ub=ub,
                                                           name=f"reverse_constraint_{reaction.id}")
    model.solver.add(reverse_constraint)


def _imat_neg_weight_(model: cobra.Model, rxn: str, threshold: float) -> None:
    """
    Internal method for adding negative weight constraints to the model.

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn: The reaction ID to add the constraint to.
    :type rxn: str
    :param threshold: The threshold value to use for iMAT (default: 1e-4). Represents the maximum flux for a reaction
        to be considered off.
    :type threshold: float
    :return: None
    """
    reaction = model.reactions.get_by_id(rxn)
    lb = reaction.lower_bound
    ub = reaction.upper_bound
    reaction_flux = reaction.forward_variable - reaction.reverse_variable
    y_pos = model.solver.interface.Variable(f"y_pos_{reaction.id}", type="binary")
    model.solver.add(y_pos)
    forward_constraint = model.solver.interface.Constraint(reaction_flux - ub * (1 - y_pos) - threshold * y_pos, ub=0,
                                                           name=f"forward_constraint_{reaction.id}")
    model.solver.add(forward_constraint)
    reverse_constraint = model.solver.interface.Constraint(reaction_flux - lb * (1 - y_pos) + threshold * y_pos, lb=0,
                                                           name=f"reverse_constraint_{reaction.id}")
    model.solver.add(reverse_constraint)

# endregion: Internal Methods
