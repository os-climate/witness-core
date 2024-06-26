## Carbon Emissions

This model is partly based on the latest version of DICE[^1] model.
Total C02 emissions in Gt is the sum of industrial emissions and emissions from land use change:
$$E_t = E_{t\,Ind} + E_{t\,Land}$$
with $E_{t\,Ind}$ the industrial emissions and $E_{t\,Land}$ the emissions from land use at period $t$.

### Industrial Emissions

Economic activity produces emissions, industrial emissions are:
$$E_{t\,Ind} = E_{t\, Energy} + E_{t\, Non\,Energy}$$ with $E_{t\, Energy}$ the emissions from energy use in industrial activity at $t$, $E_{t\, Non\,Energy}$ the other emissions from industrial activity.

#### Non Energy Emissions

Non energy emissions are computed as follow:
$$E_{t\, Non\,Energy} = \sigma_t (1 -\alpha_E - \beta_L)Y_t$$
with $\alpha_E$ the share of energy emissions in industrial activity ($energy\_emis\_share$), $\beta_L$ the share of emissions coming from land-use change ($land\_use\_change$), $Y_t$ the gross output and $\sigma$ the carbon intensity of the economy.
The carbon intensity of the economy is assumed to decline due to improvements in energy efficiency. Using DICE[^1] equation it evolves following:
$$\sigma_t = \sigma_{t-1} e^{\sigma_{g\, t-1}}$$
where $$\sigma_{g\, t-1} = \sigma_{g\, t-2}(1+ \sigma_{d1})^{time\,step}$$ and $\sigma_{d1}$ the decline rate of decarbonization ($decline\,rate\,decarbonization$).

#### Energy Emissions

Energy Emissions is an input coming from the energy model.

### Land emissions

Land emissions are from agriculture mix process. It contains emissions from forests (CO2 absorption by trees and CO2 emissions from deforestation)
and emissions from crops (CO2 balance emissions from farming).

### CO2 objective

A "CO2 objective" is calculated and set as an output of the model, based on the CO2 emissions:
$$CO2_{objective} =  \frac{\beta * (1 - \alpha) * \sum CO2_{emissions} }{(CO2_{emissions}^{ref} * \Delta_{years})}$$

where $CO2_{emissions}^{ref}$ is a reference value used to normalize the value of the objective, $\beta$ is a trade variable between the objectives based on the CO2 emissions or concentration and $\alpha$ is the global tradeof variable between global warning and the economy.

[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.
