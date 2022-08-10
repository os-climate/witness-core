## Industrial Emissions

This model is partly based on the latest version of DICE[^1] model.   

Economic activity produces emissions, industrial emissions are: 
$$E_{t\,Ind} = E_{t\, Non\,Energy}$$ with $E_{t\, Non\,Energy}$ the other emissions from industrial activity. 

The emissions from energy use in industrial activity are not included for now. 
  
#### Non Energy Emissions
Non energy emissions are computed as follow: 
$$E_{t\, Non\,Energy} = \sigma_t (1 -\alpha_E - \beta_L)Y_t$$
with $\alpha_E$ the share of energy emissions in industrial activity ($energy\_emis\_share$), $\beta_L$ the share of emissions coming from land-use change ($land\_use\_change$), $Y_t$ the gross output and $\sigma$ the carbon intensity of the economy. 
The carbon intensity of the economy is assumed to decline due to improvements in energy efficiency. Using DICE[^1] equation it evolves following: 
$$\sigma_t = \sigma_{t-1} e^{\sigma_{g\, t-1}}$$
where $$\sigma_{g\, t-1} = \sigma_{g\, t-2}(1+ \sigma_{d1})^{time\,step}$$ and $\sigma_{d1}$ the decline rate of decarbonization ($decline\,rate\,decarbonization$). 


[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.

