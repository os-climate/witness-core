## Industry economic sector model

The industry sector represents one of the identified main economic sectors. In this model we compute the evolution of the economic activity for this sector (capital, production) taking in input investment, energy and workforce available for this sector.

### Main Inputs

- Damage data ($damage\_df$): Dataframe with damage fraction to be applied to output
- Sector workforce ($workforce_df$): Dataframe with workforce in the sector per year in million of people
- Sector energy quantity ($energy\_production$): Dataframe with Total Final Consumption of energy per year in Pwh for the sector
- Sector investment ($sector\_investment$): Dataframe with investment in sector per year in 1e12\$
- Damage to productivity ($damage\_to\_productivity$): If True: apply damage to productivity. if False: Apply damage only to production.

### Outputs

- Capital detailed df ($capital\_detail\_df$): Dataframe with all variables relative to capital calculation (energy efficiency, E max, usable capital and capital) per year
- Capital ($capital\_df$): Dataframe with coupling model outputs from previous dataframe. It contains capital and usable capital in 1e12 \$ per year
- Production dataframe ($production\_df$): Dataframe with sector output per year in 1e12\$
- Productivity df ($productivity\_df$): Dataframe with productivity and productivity growth rate (in case of no climate damage) per year.

### Time Step

The time step $t$ in each equation represents the period we are looking at. In the inputs we initialize the data with 2020 information.

### Global output

#### Usable capital

Global output is calculated from a production function. Here, it is a different one from DICE model[^1] (Nordhaus, 2017) because we want to include energy as a key element in the production process. One way to do so is by directly including energy production as production factor in the production function. We chose a different option as we wanted to take this aspect into account but not to under consider the importance of labor and capital in the production process. It is the combination of capital and energy that generates the most production. Capital without energy is almost useless. It is mandatory to feed the capital with energy for it to be able to produce output such as having fuel for trucks to transport goods, or electricity for robots in factories.
For this reason the notion of usable capital ($Ku$) has been introduced that depends on the capital ($K$) and the net energy output ($En$).
Moreover, the capital is not able to absorb more energy that it is built for, thus the notion of maximum usable energy of capital ($E\_max\_k$) is also introduced.
$$Ku=K \cdot \frac{En}{E\_max\_k}$$
with $Kne$ non energy capital stock in trillions dollars (see capital section for more explaination) and $En$ the net energy supply in TWh.
The maximum usable energy of capital ($E\_max\_k$) energy evolves with technology evolution as well as the productivity of the capital ($P$):
$$E\_max\_k = \frac{K}{capital\_utilisation\_ratio \cdot P}$$
with $capital\_utilisation\_ratio$ the capital utilisation rate and P the productivity of the capital represented by a logistic function:
$$P = min\_value+ \frac{L}{1+e^{-k(year-xo)}}$$
with L is $energy\_eff\_max$ in the inputs, $min\_value$ is $energy\_eff\_cst$, $xo$ is $energy\_eff\_xzero$, and $k$ $energy\_eff\_k$.

#### Gross Output

From the definition of the usable capital ($Ku)$ a standard constant elasticity of substitution (CES) function from classical economy is used to compute the GDP ($Y$):
$$Y = A \cdot (\alpha \cdot Ku^{\gamma} + (1-\alpha) \cdot L^\gamma)^{\frac{1}{\gamma}}$$
$A$ the Total Factor Productivity (TFP), $L$ the labor force in million of people $\alpha \in (0,1)$ the share parameter reflecting the capital intensity in production, $\gamma$ the substitution parameter. $\gamma = \frac{\sigma-1}{\sigma}$ where $\sigma$ is the elasticity of substitution between capital and labor.

#### Net output

Net output $Q$ is the output net of climate damage:
$$Q_t = (1- \Omega_t )Y_t$$
with $\Omega$ is the damage fraction of output explained in the documentation of the damage model.

### Productivity

The Total factor productivity (TFP) measures the efficiency of the inputs in the production process. The initial values of the productivity and productivity growth rate are obtained during the fitting of the production function. For the TFP we have 2 options:

- The standard DICE ($damage\,to\,productivity$ = $False$) where $A_t$ evolves according to:
  $$A_t = \frac{A_{t-1}}{1-A_{gt-1}}$$ with $A_g$ the productivity growth rate.
  The initial level $A_0$ can ben changed in the inputs ($productivity\_start$),
  $$A_{gt}=A_{g0} \cdot exp(-\Delta_a \cdot (t-1))$$
  and $\Delta_a$ is the percentage growth rate of $A_g$.
- The “Damage to productivity growth” one ($damage\,to\,productivity$ = $True$) comes from Moyer et al. (2014) [^4]. It applies a fraction of damage $f$ ($frac\_damage\_prod$) to the productivity instead of all damage being applied to output:
  $$A^*_t=(1-f\Omega_t) \cdot \frac{A^*_{t-1}}{1-A_{gt-1}}$$ with $A_0 =A^*_0$.
  and then damage to output $\Omega_{yt}$ becomes:
  $$\Omega_{yt} = 1- \frac{1- \Omega_t}{1-f\Omega_t}$$
  such that the output net of climate damage is
  $$Q^*_t = (1-\Omega_{yt}) \cdot Y_t \cdot (Ku_t, L_t)$$

### Capital

The capital equation is:
$$K_t = I_t + (1- \delta )K_{t-1}$$
with $I_t$ the investment in trillions dollars, and $\delta$ the depreciation rate. Each period the capital stock increases with new investment and decreases with depreciation of past period capital.

### Notes on the fitting of the production function

To obtain the value of the production function parameters we fitted our calculated production to historical data from IMF[^5] of GDP PPP (Purchasing Power Parity) in current US dollars that we calibrated to be in constant 2020 US dollars using the GDP deflator. We also used data from the IMF[^6] for the capital stock value, for population we took data from the World Bank databank[^7] and lastly for energy we used Total Final Consumption from International Energy Agency[^10].

### Other inputs

- Year start, year end and time step
- Parameters for production function: output_alpha, output_gamma
- parameters for productivity function: productivity_start, productivity_gr_start, decline_rate_tfp
- Usable capital parameters: capital_utilisation_ratio, $energy\_eff\_k$, $energy\_eff\_cst$, $energy\_eff\_xzero$, $energy\_eff\_max$
- Capital depreciation rate
- Productivity damage fraction: Fraction of damage applied to productivity
- Initial output growth rate

## References

[^4]: Moyer, E. J., Woolley, M. D., Matteson, N. J., Glotter, M. J., & Weisbach, D. A. (2014). Climate impacts on economic growth as drivers of uncertainty in the social cost of carbon. The Journal of Legal Studies, 43(2), 401-425.

[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.

[^5]: International Monetary Fund. (2020). World Economic Outlook Database. Available at: <https://www.imf.org/en/Publications/WEO/weo-database/2020/October>

[^6]: International Monetary Fund. (2019) Investment and Capital Stock Dataset.

[^7]: World Bank.[World data bank: https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.MKTP.KD&country=](https://data.worldbank.org/)

[^10]: IEA 2022; World total final consumption by source, <https://www.iea.org/reports/key-world-energy-statistics-2020/final-consumption>, License: CC BY 4.0.
