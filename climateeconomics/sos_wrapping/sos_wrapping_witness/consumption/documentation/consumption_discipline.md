## Consumption model

As in DICE (Nordhaus, 2017[^1]), this model is used for the optimisation problem. Our way to compute utility per capita is slightly different from DICE. We multiply the value obtained in DICE by an energy price ratio and a residential energy ratio.
Consumption is:
$$C_t = Y_t - I_t$$
with
$Y_t$ is the net output (GDP) coming from Macroeconomics, and $I_t$ is the total investment also coming from macroeconomics model.

Utility per capita is:
$$U_{pc}(c_t) = energy\_price\_ratio_t.residential\_energy\_ratio_t.(\frac{c_t^{1-\alpha}}{1-\alpha}-1)$$
with
$$energy\_price\_ratio_t =\frac{energy\_price_{t=0}}{energy\_price_t}$$
and
$$residential\_energy\_ratio_t =\frac{residential\_energy_t}{residential\_energy_{2019}}$$
where $energy\_price$ is the mean price of energy in dollars per MWh and comes from the energy model, $residential\_energy$ is the residential energy coming from energy mix, $\alpha$ is the elasticity of marginal utility ($conso\_elasticity$). We set it by default to 2. A higher $\alpha$ means that marginal utility decreases faster with increase in income. $c_t = \frac{C_t}{L_t}$ is the per capita consumption. $L_t$ is the population in millions of people.

### Objectives

The DICE's objective function is the social welfare, the discounted sum of utility:
$$W = \sum_{t=1}^{t_{max}}U_{pc}(c_t)R_tL_t$$
with $R_t$ the discount factor, a discount on the economic well-being of future
generations.
$$R_t = \frac{1}{(1+\rho)^t}$$
and $\rho$ is the pure rate of social time preference.
The discount rate is a very debated parameter as it can change drastically the results and no consensus exists on its value. This topic has been extensively analysed by Gaullier (2011)[^2].

Another objective value is calculated and set as an output of the utility model, based on the minimum value of the discounted utility:

$$ utility*{min} = \alpha * (1 - \gamma) * utility*{min}^{ref} / min(U\_{pc}(c_t)R_tL_t)$$

where $\gamma$ is a trade variable between the objectives based on the economy and $\alpha$ is the global tradeof variable between global warning and the economy.

### References

[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.

[^2]: Gollier, C. (2011). Pricing the future: The economics of discounting and sustainable development. Unpublished Manuscript, to Appear with Princeton University Press, Princeton, NJ, USA.
