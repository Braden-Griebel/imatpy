Usage
=====

Installation
************

iMATpy can be installed with pip: ``pip install imatpy``
  
iMATpy requires an optimizer to be installed. The default optimizer is ``glpk`` which is installed by ``COBRApy``. This package can also be used with Gurobi_ or CPLEX_ 
which must be installed seperately. Both ``cplex`` and ``gurobi`` have free academic licenses. Once installed, the optimizer can be changed by setting the solver property 
of a model, and the default can be changed via ``COBRApy`` config, by changing the ``solver`` property of the ``cobra.core.configuration.Configuration()`` object (see `cobrapy documentation`_). 

.. _Gurobi: https://www.gurobi.com/
.. _CPLEX: https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer
.. _cobrapy documentation: https://cobrapy.readthedocs.io/en/latest/

Package Structure
*****************
The package is divided into four main submodules, ``imat``, ``model_creation``, ``parse_gpr``, and ``model_utils``. 

- ``imat``: submodule containing functions for running the iMAT algorithm and generating flux distributions. 
- ``model_creation``: submodule contains functions for creating models based on iMAT solutions. Doesn't require using any of the `imat` functions seperately,
  as they are called internally as needed for the different model creation methods. 
- ``parse_gpr``: submodule contains functions for using the gene protein reaction rules in a model for converting gene expression weights into reaction weights.
  The gene expression data should already be converted into weights before using these functions (and so should only contain -1, 0, and 1). The functions assume a
  single series to be used, so to use multiple observations, the data should be combined (e.g. by taking the mean or median) before using these functions.
- ``model_utils``: submodule contains various utility functions for working with models. This includes helper functions for reading a writing models to various formats,
  as well as functions for checking if two models are equivalent.   



Examples
********
.. 
