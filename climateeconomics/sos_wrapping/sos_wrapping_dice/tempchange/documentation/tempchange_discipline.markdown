## Temperature model
The temperature model computes the evolution of atmospheric and ocean temperature following change in C02 concentration. In contains three equations, one for the radiative forcing and two equation for the climate system, all extracted from DICE[^1]  model with no modification (Nordhaus, 2017). 

### Time step 
The time step $t$ in each equation represents the period we are looking at. In the inputs we initialise the data with 2020 information. The user can choose the year end and the duration of the period (in years) by changing the parameters $year\, end$ and $time \,step$. For example, for a year start at 2020, year end in 2100 and a duration of time step of 5 years we have $t \, \epsilon \,[0, 16]$.

### Radiative forcing
The radiative forcing equation computes the impact of the accumulation of GHGs on the radiation balance of the globe. The change in total radiative forcings of GHGs from antrhopogenic sources is: 
$$F_t = \eta.\frac{\ln(M_{t\,AT}) - ln(M^{1750}_{AT})}{ln(2)}+F_{t\,EX}$$
$F_t$ in watts per m2, $M_{t\,AT}$ the atmospheric concentration of C02 in Gtc (defined in Carbon Cycle model), $M^{1750}_{t\,AT}$ the level of atmospheric concentration before industrial revolution, $\eta$ the increase in forcing from the doubling of CO2 in the atmosphere and $F_{t\,EX}$ forcing from gases other than CO2 at $t$. 
If $t < t_{max}$:  
 $$F_{t\,EX}= F_{0\,EX} + \frac{1}{17}(F_{t_{max}\,EX} - F_{0\,EX}).(t-1)$$
 and for $t = t_{max}:\:F_{t\,EX}= F_{t_{max}\,EX}$ corresponding to $hundred\_forcing\_nonco$ in the inputs and $t_{max}$ the last period. 


### Atmospheric temperature
Radiative forcing warms up the atmosphere which leads to a warm up in the upper ocean and then the lower ocean. The increase in atmospheric temperature in degree celsius is:  
$$T_{t\,AT} = T_{t-1\,AT} + \xi_1[F_t - \lambda T_{t-1} - \xi_2(T_{t-1} - T_{t-1\,LO})]$$
with $\xi_i$ the transfer coefficients reflecting the rates of flow and thermal capacities of the sinks for a $time step$ = 5 that we adapt to the current time step. $\xi_1$ is $climate\_upper$ in the inputs and $\xi_2$ $transfer\_upper$. Lastly $\lambda  = \frac{forcing\_eq\_C02}{eq\_temp\_impact}$ is climate sensitivity.

### Ocean temperature
The increase in ocean temperature is: 
$$T_{t\,LO} = T_{t-1\,LO} + \xi_3(T_{t-1} - T_{t-1\,LO})$$
with $\frac{1}{\xi_3}$ the transfer rate from the upper ocean to the deep ocean ($transfer\_lower$ in the inputs).   

### References 

[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.