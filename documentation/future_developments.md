Antoine : note for future developments:

##  Agriculture sector
- handle demand vs production of biomass
- compute ratios
- connect the ratios to energy technos that consumes biomass

## Optimization and gradients :
Auto differentiability is great, but its long to run cause there is no intelligence for the moment,
its brutforcing the computation for every output wrt every input.
in can be sped up by
- saving gradients in a cache linked to the model when the gradients are constant (identity, zeros)
- only using the autodiff computation when gradients are complex

Then we should re-activate all the gradients testing.



