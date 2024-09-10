## Carbon cycle

For the carbon cycle model we use the one from DICE (Nordhaus, 2017[^1]) without any modification.
It uses a three-reservoir model calibrated with historical data. The three reservoirs are the deep ocean, the upper ocean and the atmosphere. Carbon flows in both directions between the reservoirs. $M_{t\,i}$ represents the mass of carbon in gigatons in the reservoir $i$ at period $t$:
$$M_{t+1\,AT} = M_{t\,AT}.b11 + M_{t\,UP}.b21 + E_t.\frac{time\,step}{3.666}$$
$$M_{t+1\,UP}= M_{t\,AT}.b12 + M_{t\,UP}.b22 + M_{t\,LO}.b32$$
$$M_{t+1\,LO}= M_{t\,LO}.b33 + M_{t\,UP}.b23$$

with $AT$ the atmosphere reservoir, $UP$ the upper ocean and $LO$ the deeper ocean. The parameters $b_{ij}$ represents the flows between the reservoirs.

The model only considers CO2, other greenhouse gases are assumed to be exogenous and enter the forcing equation explained in $Temperature$ model.


## References
[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.
