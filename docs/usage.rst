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



Workflow
********


1. Normalize gene expression data for sequencing depth and gene length (TPM or RPKM both work well)
2. Quantile normalize the gene expression data (for example with the `qnorm package`_) (This is optional, but can help make the results less sensitive to parameters)
3. Combine gene expression values accross samples, for example by taking the mean or median
4. Convert the gene expression data into qualitative weights, of -1, 0, or 1, where -1 represents low expression, 1 represents high expression, and 0 represents all other genes. This can be done by setting the bottom 15% of genes to -1, the top 15% of genes to 1, and all other genes to 0. Other percentiles can be used, but 15% is a decent starting point.
5. Using the `parse_gpr` submodule, specifically the `gene_to_rxn_weights` function, convert the gene weights into reaction weights using the GPR rules in the model.
6. Either generate a flux distribution with the iMAT algorithm, or construct a new model with restrictions based on iMAT results.

    - To generate a flux distribution, use the `imatpy.imat` submodule, specifically the `imat` function. This function takes a model, and a series of reaction weights, and returns a flux distribution which maximizes the sum of reactions with high expression which have a flux above `epsilon`, and reactions with low expression which have a flux below `threshold`.
    - To generate a model with restrictions based on the iMAT results, use the `model_creation` submodule. It includes a variety of different methods for generating models based on iMAT, with a wrapper function `generate_model` for convinience.

.. _qnorm package: https://pypi.org/project/qnorm/

Parse GPR
*********
The ``parse_gpr`` submodule has functions for applying the gene-protien-reaction (GPR) rules in a model to gene weights in order to convert gene weights into reaction weights which can act as input to the iMAT algorithm.

The input to this method is a pandas Series with the index being the gene identifiers, and the values being the gene weights. The output is a pandas Series with the index being the reaction identifiers, and the values being the reaction weights. This method only handles a single series, so the gene expression data must be processed into single weights (with -1 representing lowly expressed genes, 1 representing highly expressed genes, and 0 for all other genes).

Here is an example of how to use this method:

.. exec_code::

   # External imports
   from cobra.core.configuration import Configuration
   import pandas as pd

   # iMATpy imports
   from imatpy.model_utils import read_model
   from imatpy.parse_gpr import gene_to_rxn_weights

   # Set the solver to glpk
   Configuration().solver = "glpk"

   # Read in the model
   model = read_model("./tests/data/test_model.xml")

   # Create a pandas Series representing gene expression weights
   model_weights = pd.Series({
              "g_A_imp": 1,
              "g_B_imp": -1,
              "g_C_imp": -1,
              "g_F_exp": 0,
              "g_G_exp": -1,
              "g_H_exp": 0,
              "g_A_B_D_E": 0,
              "g_C_E_F": -1,
              "g_C_H": 0,
              "g_D_G": 1,
          })

   # Convert the gene weights into reaction weights
   reaction_weights = gene_to_rxn_weights(model=model, gene_weights=model_weights)

   # Print the reaction weights
   print(reaction_weights)

iMAT methods
************
The ``imat`` submodule contains functions for running the iMAT algorithm. The main function is ``imat``, which takes a model, and a series of reaction weights, and returns a flux distribution which maximizes the sum of reactions with high expression which have a flux above ``epsilon``, and reactions with low expression which have a flux below ``threshold``.

Here is an example of how to use this method:

.. exec_code::

   # External imports
   from cobra.core.configuration import Configuration
   import pandas as pd

   # iMATpy imports
   from imatpy.model_utils import read_model
   from imatpy.imat import imat

   # Set the solver to glpk
   Configuration().solver = "glpk"

   # Read in the model
   model = read_model("./tests/data/test_model.xml")

   # Read in the reaction weights
   rxn_weights = pd.read_csv("./tests/data/test_model_reaction_weights.csv", index_col=0).squeeze()

   # Run iMAT
   imat_results = imat(model=model, rxn_weights=rxn_weights, epsilon=1, threshold=0.01)

   # Print the imat objective
   print(f"iMAT Objective: {imat_results.objective_value}")

   # Print the imat flux distribution
   print(f"iMAT Flux Distribution: \n{imat_results.fluxes}")

Model Creation
**************

The ``model_creation`` submodule contains functions for creating new models based on the results of iMAT. The main function is ``generate_model``, which takes a model, and a series of reaction weights, and returns a new model with restrictions based on the iMAT results.

The available methods for creating a model based on an iMAT flux distribution is:

``imat_restrictions``
  Adds the binary variables and constraints used in the iMAT algorithm, as well as an additional
  constraint ensuring that the flux distribution is within tolerance of the optimal iMAT objective
  value. This method stays closest to the iMAT objective, but the included indicator (binary)
  variables mean that is unsuitable for sampling.

``simple_bounds``
  Adds bounds on the reactions found to be "on", and "off" in iMAT. For all the highly
  expressed reactions found to be "on", the flux is constrained to be at least ``epsilon``.
  For all the lowly expressed reactions found to be "off", the flux is constrained to be
  below ``threshold``.

``subset``
  Removes reactions from the model which are found to be "off". For all the lowly expressed
  reactions found to be off, they are constrained to have a flux below ``threshold``.

``fva``
  Finds bounds using an FVA like approach. A temporary model is created in a simmilar way to the
  ``imat_restrictions`` method above, which includes the imat variables, constraints, and which also
  constrains the flux distribution to be near optimal for iMAT. The maximum and minimum fluxes
  allowed through each reaction (while still maintaining the optimal iMAT objective) is found.
  These values are used as the new reaction bounds. It should be noted, that although the individual
  upper and lower bounds for the reaction are achievable for each reation while being consistant
  with the optimal iMAT objective, this doesn't guarantee that the flux distribution overall is
  consistant with the optimal iMAT objective.

``milp``
  Uses a set of mixed integer linear programs to find whether a reaction should be forced
  off, forward, or reverse. Each reaction in turn is forced to be off, active in the forward
  direction, and active in the reverse direction, and the iMAT objective is maximized. Whether
  a reaction should be forced off, or active in either the forward or reverse direction is then
  determined by which direction maximizes the iMAT objective. Again, it should be noted that
  this doesn't guarantee that the iMAT objective is overall maximized by solutions to this model.

Here is an example of how to use this method:

.. exec_code::

   # External imports
   from cobra.core.configuration import Configuration
   import pandas as pd

   # iMATpy imports
   from imatpy.model_utils import read_model
   from imatpy.model_creation import generate_model

   # Set the solver to glpk
   Configuration().solver = "glpk"

   # Read in the model
   model = read_model("./tests/data/test_model.xml")

   # Read in the reaction weights
   rxn_weights = pd.read_csv("./tests/data/test_model_reaction_weights.csv", index_col=0).squeeze()

   # Generate a new model based on the iMAT results
   new_model = generate_model(model=model, rxn_weights=rxn_weights, method="fva")

Model Utils
***********

The ``model_utils`` submodule contains several utility functions for working with COBRApy models. Specifically, it contains functions for:
- Reading and writing models in various formats with a single function, specifically ``read_model``/``write_model``.
- Determining if two models are equivalent, using ``model_eq``.

Here is an example of how to use the model IO methods:

.. code-block::python
  # iMATpy imports
  from imatpy.model_utils import read_model, write_model, model_eq

  # You can read in a model from a file
  model = read_model("./tests/data/test_model.xml") # in SBML
  model = read_model("./tests/data/test_model.json") # in JSON
  model = read_model("./tests/data/test_model.yml") # in YAML
  model = read_model("./tests/data/test_model.mat") # in Matlab

  # You can also write a model to a file
  write_model(model, "./tests/data/test_model.xml") # in SBML
  write_model(model, "./tests/data/test_model.json") # in JSON
  write_model(model, "./tests/data/test_model.yml") # in YAML
  write_model(model, "./tests/data/test_model.mat") # in Matlab

Here is an example of using the model comparison method:

.. exec_code::

   # External imports
   from cobra.core.configuration import Configuration

   # iMATpy imports
   from imatpy.model_utils import read_model, model_eq

   # Set the solver to glpk
   Configuration().solver = "glpk"

   # Read a model
   model = read_model("./tests/data/test_model.xml")

   # Create a copy of the model
   model_copy = model.copy()

   # Check that the models are equivalent
   print(f"Models are equivalent: {model_eq(model, model_copy)}")

   # Change the copy model
   model_copy.reactions.get_by_id("r_A_B_D_E").lower_bound = -314

   # Check that the models are no longer equivalent
   print(f"Models are equivalent: {model_eq(model, model_copy)}")
