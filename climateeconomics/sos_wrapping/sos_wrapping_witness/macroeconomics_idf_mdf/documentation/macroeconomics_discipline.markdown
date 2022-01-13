## Macroeconomics model
### Time Step 
The time step $t$ in each equation represents the period we are looking at. In the inputs we initialize the data with 2020 information. The user can choose the year end and the duration of the period (in years) by changing the parameters $year\, end$ and $time \,step$. For example, for a year start at 2020, year end in 2100 and a duration of time step of 5 years we have $t \, \epsilon \,[0, 16]$.


### Global output
#### Gross output 
Global output is calculated from a production function. Here, it is a different one from DICE model[^1] (Nordhaus, 2017) because we include Energy as production factor. We created our own function by fitting historical data. The equation for the global gross output is: 
$$economic\_part_t = A_t(c(pop\_factor * L_t)^\alpha + (1-c)K_t^\beta)$$
$$energy\_part_t = A^E_t * (energy\_factor * E_t)^\gamma$$
$$Y_t = (b*economic\_part_t + (1-b)* energy\_part_t)^\theta$$
with $t$ the time step, $K$ the capital stock in trillions dollars, $L$ the population in millions of people, $E$ the energy supply in TWh, $A$ the productivity and $A^E$ the energy productivity. $\alpha$, $\beta$, $\gamma$, $\theta$, $b$ and $c$ are obtained by fitting of historical data. $K_t$ and $L_t$ calculation is explained below and the energy supply is an output of the energy model. 

#### Net output 
Net output $Q$ is the output net of climate damage:
$$Q_t = (1- \Omega_t )Y_t$$
with $\Omega$ is the damage fraction of output explained in the documentation of the damage model.  

### Productivity
The Total factor productivity (TFP) measures the efficiency of the inputs in the production process. The initial values of the productivity and productivity growth rate are obtained during the fitting of the production function. For the TFP we have 2 options: 
* The standard DICE ($damage\,to\,productivity$ = $False$) where $A_t$ evolves according to:
$$A_t = \frac{A_{t-1}}{1-A_{gt-1}}$$ with $A_g$ the productivity growth rate.
The initial level $A_0$ can ben changed in the inputs ($productivity\_start$),
$$A_{gt}=A_{g0}exp(-\Delta_a(t-1)time\_step)$$
and $\Delta_a$ is the percentage growth rate of $A_g$.
* The “Damage to productivity growth” one ($damage\,to\,productivity$ = $True$) comes from Moyer et al. (2014) [^4]. It applies a fraction of damage $f$ ($frac\_damage\_prod$) to the productivity instead of all damage being applied to output:
$$A^*_t=(1-f\Omega_t)\frac{A^*_{t-1}}{1-A_{gt-1}}$$ with $A_0 =A^*_0$.  
and then damage to output $\Omega_{yt}$ becomes: 
$$\Omega_{yt} = 1- \frac{1- \Omega_t}{1-f\Omega_t}$$
such that the output net of climate damage is 
$$Q^*_t = (1-\Omega_{yt})Y_t(K_t, L_t, A_t^{*E}, A^*_t, E_t)$$

### Energy productivity 
The energy productivity measures the efficiency of the energy supply input in the production process. It can also represent the inverse of the energy intensity of the economy. For this variable we use the same equations as for the classical productivity. The initial values of the energy productivity and energy productivity growth rate are obtained during the fitting of the production function. We also have two options: 
* The standard ($damage\,to\,productivity$ = $False$) where $A_t^E$ evolves according to:
$$A_t^E=\frac{A_{t-1}^E}{1-A_{gt-1}^E}$$
with $A_g^E$ the productivity growth rate.
The initial level $A_0^E$ can ben changed in the inputs ($init\_energy\_productivity$),
$$A_{gt}^E=A_{g0}^Eexp(-\Delta_a^E(t-1)time\_step)$$
and $\Delta_a^E$ is the percentage growth rate of $A_g^E$.
* The “Damage to productivity growth” one ($damage\,to\,productivity$ = $True$) comes from Moyer et al. (2014) [^4]. It applies a fraction of damage $f$ ($frac\_damage\_prod$) to the productivity instead of all damage being applied to output:
$$A^{*E}_t=(1-f\Omega_t)\frac{A^{*E}_{t-1}}{1-A_{gt-1}^E}$$
with $A_0 =A^*_0$.  
and then damage to output $\Omega_{yt}$ becomes: 
$$\Omega_{yt} = 1- \frac{1- \Omega_t}{1-f\Omega_t}$$
such that the output net of climate damage is 
$$Q^*_t = (1-\Omega_{yt})Y_t(K_t, L_t, A_t^{*E,} A^*_t, E_t)$$ 

### Capital
The capital equation is: 
$$K_t = I_t + (1- \delta )K_{t-1}$$
with $I_t$ the investment in trillions dollars, and $\delta$ the depreciation rate. Each period the capital stock increases with new investment and decreases with depreciation of past period capital.

### Investment
Investment is defined using the inputs $share\_energy\_investment$ and $share\_non\_energy\_investment$.

The investment in energy $I^E$ is: $$I_{t}^E = share\_energy\_investment_t * Q_t + ren\_investments$$
With:
$$ren\_investments = emissions * co2\_taxes * co2\_tax\_eff$$
However, invest coming from CO2 taxes are capped at the value of energy investment without tax multiplied by the model input factor co2_input_limit. It is 2 by default and smothed with the following formula:
$$ren\_investments = co2\_invest\_limit * energy\_investment\_wo\_tax / 10.0 * \\ (9.0 + exp(- co2\_invest\_limit * energy\_investment\_wo\_tax / ren\_investments))$$

The investment in non-energy $I^{NE}$ is :  $$I_{t}^{NE} = share\_non\_energy\_investment_t * Q_t$$ 
and the total investment $I_t =  I_t^E + I_t^{NE}$ is limited to a certain share of the net output set by $max\_invest$ input. 

### Consumption
Consumption is such that: 
$$C_t = Y_t - I_t$$
The part of the output not invested is used for consumption. 

### Population
Population is a static population forecast dataframe from GHDx[^8] in million of persons.
The population data used is the Reference curve on the graph below.
![](population.png)


### Notes on the fitting of the production function
To obtain the value of the production function parameters we fitted our calculated production to historical data from IMF[^5] of GDP PPP (Purchasing Power Parity) in current US dollars that we calibrated to be in constant 2020 US dollars using the GDP deflator. We also used data from the IMF[^6] for the capital stock value and for population we took data from the World Bank databank[^7].

## References

[^4]: Moyer, E. J., Woolley, M. D., Matteson, N. J., Glotter, M. J., & Weisbach, D. A. (2014). Climate impacts on economic growth as drivers of uncertainty in the social cost of carbon. The Journal of Legal Studies, 43(2), 401-425.

[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.

[^5]: International Monetary Fund. (2020). World Economic Outlook Database. Available at: https://www.imf.org/en/Publications/WEO/weo-database/2020/October

[^6]: International Monetary Fund. (2019)  Investment and Capital Stock Dataset.


[^7]: World Bank.[ World data bank: https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.MKTP.KD&country=](https://data.worldbank.org/)

[^8]: GHDx, (2020) Global Fertility, Mortality, Migration, and Population Forecasts 2017-2100, Available at: http://ghdx.healthdata.org/record/ihme-data/global-population-forecasts-2017-2100