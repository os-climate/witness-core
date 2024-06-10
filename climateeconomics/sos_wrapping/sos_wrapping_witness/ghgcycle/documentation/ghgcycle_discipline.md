## Greenhouse Gas cycle

For the Greenhouse Gas cycle model we use the one from FUND.[^4]

Methane, nitrous oxide and sulphur hexafluoride are taken up in the atmosphere, and then geometrically depleted:

$$C_{t} = C_{t-1} + \alpha E_{t} - \beta (C_{t-1} - C_{pre}) $$

where $C$ denotes concentration, $E$ emissions, $t$ year, and $\text{pre}$ pre-industrial. $\alpha$ and $\beta$ are given parameters for all gases.
Parameters are taken from Forster et al. (2007).[^5]

The atmospheric concentration of carbon dioxide follows from a five-box model:

$$Box_{i, t} = \rho_i Box_{i, t-1} + 0.000471\alpha_i E_t$$

with

$$C_t = \sum_{i=1}^5 \alpha_i Box_{i, t}$$

where $\alpha_{i}$ denotes the fraction of emissions $E$ (in million metric tonnes of carbon) that is allocated to $Box_{i}$ ($0.13$, $0.20$, $0.32$, $0.25$ and $0.10$, respectively) and $\rho$ the decay-rate of the boxes $(\rho = exp( - \frac{1}{\mathrm{\text{lifetime}}})$
with life-times $infinity$, $363$, $74$, $17$ and $2$ years, respectively).

The model is due to Maier-Reimer and Hasselmann (1987)[^1].
Its parameters are due to Hammitt et al. (1992)[^2]. Thus, $13 \%$ of total emissions remains forever in the atmosphere, while $10\%$ is — on average — removed in two years. Carbon dioxide concentrations are measured in parts per million by volume.
You can find an implementation (and parameters) of this model on GitHub.[^3]

### Model inputs and outputs

The inputs are:

- year start, the starting year of the study. Default is 2020.
- year end, the last year of the study. Default is 2100.
- time step, the number of year between each step of computation. Default is 1.
- GHG emissions, the quantity of GHG released in the atmosphere in Gt. This data comes from ghg_emission model.
- Alpha is the trade variable between utility and CO2 emission, used to compute the output ppm objective. The weight of utility is Alpha, the weight of climate is (1 - Alpha). The default value is 0.5.
- Beta is the trade variable between CO2 emission and temperature, used to compute the output ppm objective. The weight of CO2 emission is Beta, the weight of temperature is (1 - Beta).
- Numerical parameters from the FUND model. In particular, emissions fractions, decay rates and initial concentrations for CO2, CH4 and N2O gases.

The outputs of the models are:

- ppm_objective gives the concentration of carbon of the atmosphere in parts per millions for each year.
- GHG cycle data gives the concentration of carbon in the atmosphere for each year.
- GHG_cycle_detail_df gives the concentration of carbon in atmoshpere and ocean.

### PPM objective

A ppm objective is computed and set as an output of the model, based on the number of CO2 particule per meter:
$$ppm_{objective} = \frac{(1 - \alpha) \times (1 - \beta) \times \sum CO2_{ppm} }{(CO2_{ppm}^{ref} \times \Delta_{years})}$$

where $CO2_{ppm}^{ref}$ is a reference value used to normalize the value of the objective, $\beta$ is a trade variable between the objectives based on the CO2 emissions or concentration and $\alpha$ is the global tradeof variable between global warning and the economy.

## References

[^1]: Maier-Reimer, E. and K.Hasselmann (1987), 'Transport and Storage of Carbon Dioxide in the Ocean: An Inorganic Ocean Circulation Carbon Cycle Model', Climate Dynamics, 2, 63-90.

[^2]: Hammitt, J.K., R.J.Lempert, and M.E.Schlesinger (1992), 'A Sequential-Decision Strategy for Abating Climate Change', Nature, 357, 315-318.

[^3]: FUND Repository on GitHub (<https://github.com/fund-model/MimiFUND.jl/tree/master/src>)

[^4]: FUND Model Online Documentation (<http://www.fund-model.org/MimiFUND.jl/latest/science/#.-Atmosphere-and-climate-1>)

[^5]: Forster, P., V. Ramaswamy, P. Artaxo, T. Berntsen, R. Betts, D. W. Fahey, J. Haywood, J. Lean, D. C. Lowe, G. Myhre, J. Nganga, R. Prinn, G. Raga, M. Schulz and R. V. Dorland (2007). Changes in Atmospheric Constituents and in Radiative Forcing. Climate Change 2007: The Physical Science Basis. Contribution of Working Group I to the Fourth Assessment Report of the Intergovernmental Panel on Climate Change. S. Solomon, D. Qin, M. Manning et al. Cambridge, United Kingdom and New York, NY, USA, Cambridge University Press.
