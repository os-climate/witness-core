## Objectives discipline

This model takes in inputs historical data and simulated data for capital and output and computes the mean squared errors (MSE). 

$$Mean\_squared\_error = \frac{1}{N} \sum_{i=1}^{N}(x_i-y_i)^2$$
with $x_i$ the observed value and $y_i$ the predicted value. 

### Inputs 
- Economics df ($economics\_df$): a dataframe with simulated total capital and output in T$
- The production df ($production\_df$): the dataframe with output and net output in T$ for each sector in $sector\_list$. 
- The capital df ($production\_df$): the dataframe with capital and usable capital stock in T$ for each sector in $sector\_list$. 
- Sector list ($sector\_list$): the list of sector coming from the macroeconomics model
- Historical gdp ($historical\_gdp$): the historical data for the output in T$
- Historical capital ($historical\_capital$): the historical data for the capital in T$
- 
### Outputs
- $error\_pib\_total$ : the MSE for the total output
- $error\_cap\_total$: the MSE for the total capital
- $sectors\_cap\_errors$: a dictionary with the MSE for each sector capital
- $sectors\_gdp\_errors$: a dictionary with the MSE for each sector output
- year start and year end 