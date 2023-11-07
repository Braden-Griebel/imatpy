Usage
=====

Installation
************

iMATpy can be installed with pip: ``pip install imatpy``

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
TODO
