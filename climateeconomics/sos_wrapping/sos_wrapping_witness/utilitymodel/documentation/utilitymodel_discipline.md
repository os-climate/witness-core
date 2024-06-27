## Utility model
As in DICE (Nordhaus, 2017[^1]), this model is used for the optimisation problem. Our way to compute utility per capita is slightly different from DICE. We multiply the value obtained in DICE by an energy price ratio. 
Furthermore, to avoid negative values when the capital consumption is low (in case of large damages for instance), Utility per capita is: 
$$U_{pc}(c_t) = energy\_price\_ratio_t.c_t$$
with 
$$energy\_price\_ratio_t =\frac{energy\_price_{t=0}}{energy\_price_t}$$
where $energy\_price$ is the mean price of energy in dollars per MWh and comes from the energy model, $\alpha$ is the elasticity of marginal utility ($conso\_elasticity$). We set it by default to 2. A higher $\alpha$ means that marginal utility decreases faster with increase in income. $c_t = \frac{C_t}{L_t}$ is the per capita consumption. $L_t$ is the population in millions of people.

### Objective

$$\text{maximize}_{x \in \text{design space}} \text{ Population utility objective (x)}$$


The `Population utility objective` (a float) is the average over the years of utility of the population.
$$\text{Population utility objective} = \frac{1}{\text{nb years}}\sum_{\text{year in years}} \text{Population utility (year)}$$

with 

$$\text{Population utility} = \frac{\text{Population}}{\text{Population at year start}} \text{Utility per capita}$$

The next section described the notion of  *Utility per capita*.

#### Utility per capita

The utility per capita relies on two variables available in witness, *Consumption per capita* and *Energy price*. The next two sections gives a quick explanation of these variables.

In our optimization formulation, we want to maximize the quantity of things consumed. For that, we can see *Consumption per capita* can be seen as 

$$C^{pc} = Q^{pc} \times P$$

that is, a quantity (of "things" consumed) $\times$ Price ("average price of things consumed"). 
The assumption we make is that the average price of things that are consumed is driven by energy price, leading to :


$$\text{quantity per capita} = \frac{\text{consumption per capita}}{\text{energy price}}$$

If we take year start as a reference point, and apply a function $f$ to mimic saturation to consumption (having more when your poor is huge, but having more when you already have a lot doesnt mean much to you), we defined the gain of utility as

$$\text{utility per capita (year)} = f \left(\frac{\text{quantity per capita (year)}}{\text{quantity per capita (year start)}} \right)$$

> This saturation function is an S-curve, whose parameters have been fine-tuned, but can be tweeked based on your preferences. 


### References
[^1]: Nordhaus, W. D. (2017). Revisiting the social cost of carbon. Proceedings of the National Academy of Sciences, 114(7), 1518-1523.

[^2]: Gollier, C. (2011). Pricing the future: The economics of discounting and sustainable development. Unpublished Manuscript, to Appear with Princeton University Press, Princeton, NJ, USA.
