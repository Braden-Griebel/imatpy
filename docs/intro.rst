Introduction
============

``iMATpy`` is a python package for using the iMAT algorithm (Shlomi T, et al. Network-based prediction of human tissue-specific metabolism, Nat. Biotechnol., 2008, vol. 26 (pg. 1003-1010)) to integrate gene expression data with genome scale metabolic models. 

Algorithm 
*********

The iMAT algorithm is a method for finding a flux distribution from a metabolic model which matches gene expression data. The algorithm takes a genome scale metabolic model, and a set of reaction weights, and returns a feasible flux distribution that is most consistant with the provided reaction weights. It does this by maximizing the number of reactions associated with highly expressed genes which are "on" (flux above `epsilon`), while minimizing the number of reactions associated with lowly expressed genes which are "off" (flux below `threshold`). This flux distribution can then be used to update the metabolic model so that it reflects the gene expression data. 

Model Creation
**************

The iMAT algorithm returns a flux distribution which is consistant with the provided gene expression data, which is useful, but most of the time we want to be able to work with a model rather than just this vector of weights. ``iMATpy`` offers a variety of methods for creating an updated model using the iMAT algorithm via the `generate_model` function. 

`imat_restrictions`
  Adds the binary variables and constraints used in the iMAT algorithm, as well as an additional
  constraint ensuring that the flux distribution is within tolerance of the optimal iMAT objective 
  value. This method stays closest to the iMAT objective, but the included indicator (binary) 
  variables mean that is unsuitable for sampling. 

`simple_bounds`
  Adds bounds on the reactions found to be "on", and "off" in iMAT. For all the highly 
  expressed reactions found to be "on", the flux is constrained to be at least `epsilon`. 
  For all the lowly expressed reactions found to be "off", the flux is constrained to be 
  below `threshold`. 

`subset`
  Removes reactions from the model which are found to be "off". For all the lowly expressed
  reactions found to be off, they are constrained to have a flux below `threshold`. 

`fva`
  Finds bounds using an FVA like approach. A temporary model is created in a simmilar way to the 
  `imat_restrictions` method above, which includes the imat variables, constraints, and which also 
  constrains the flux distribution to be near optimal for iMAT. The maximum and minimum fluxes 
  allowed through each reaction (while still maintaining the optimal iMAT objective) is found. 
  These values are used as the new reaction bounds. It should be noted, that although the individual
  upper and lower bounds for the reaction are achievable for each reation while being consistant 
  with the optimal iMAT objective, this doesn't guarantee that the flux distribution overall is 
  consistant with the optimal iMAT objective.

`milp`
  Uses a set of mixed integer linear programs to find whether a reaction should be forced 
  off, forward, or reverse. Each reaction in turn is forced to be off, active in the forward 
  direction, and active in the reverse direction, and the iMAT objective is maximized. Whether 
  a reaction should be forced off, or active in either the forward or reverse direction is then
  determined by which direction maximizes the iMAT objective. Again, it should be noted that 
  this doesn't guarantee that the iMAT objective is overall maximized by solutions to this model. 


