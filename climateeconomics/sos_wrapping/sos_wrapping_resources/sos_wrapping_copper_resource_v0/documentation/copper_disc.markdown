The Earth offers a finite amount of gas. Theses gas stocks take different forms and according to the gas kind and the region it come from, the price varies.

### Model principle

This model takes in input the annual demand of gas resource from energy mix model (all resource demand dataframe). It gives in output the gas mean price per year, the predictable production per gas types, the quantity of gas used each year in order to answer the gas demand, and the production in excess which is stocked for the future needs.
To calculate all this dataframe the model takes in input the demand and the past production data of each gas types.
With these data we can calculate the Hubbert curves used in prediction of gas production.
The prediction on gas production shows when the addition of production and stocks will no longer be able to answer the annual demand.
Stocks are calculated with the production excess. Each year the model puts, in the stock, all the production not used in the year. 
The model uses as a priority the gas types which are the less expensive, then it stocks the more expensive one if there is production in excess.
The price is calculate as the mean value of all the extraction price. It takes each gas types price and calculate according to each gas proportion the resulting price.

### Hubbert model for production prediction [^1]

This model can give different informations : 
* The approximate year of a given resource production peak
* The estimate maximum quantity of the stock
* The projection on production capacity for the future year

Computation is based on the extrapolation of the past production. With this extrapolation we find some parameters useful in the predictable production equation:
![](production.png)

In this equation we have:
- P: production per year
- Q: cumulative production per year from year start
- tau: is the year of gas peak
- w: is the frequency 
- Qinf: the maximum stock usable 

We can find w and Qinf with the linear regression between production (P) and cumulative production (Q)

![](regression.png)

In order to find these parameters we need to find the data for the past production per year from the beginning of the production.
The used data are taken from 1950 to 2020 there is several categories of gas which represent a percentage of the total production. These proportions change with the production year. In order to get an idea of this percentage, we use a linear evolution of each gas proportion between 2000 and 2020. 

Production data sources : [^2] and [^3]

### Gas data

|Year |Conventional gas (bcm)|tight (bcm)|shale (bcm)|Coalbed methane (bcm)|other (bcm)|
| :------- | :---------- | :-----------: | :---------: | :-----------------: |:-----------------:|
|1950 |184|12|0.2|3|0.8|
|...|...|...|...|...|...|
|2020|3068|268|564|88|11|

### Fitting [^4]

To fit the curve with the maximum reserve estimate by BP we adjust the beginning year of the regression in order to take the year start at the beginning of the current peak and get realistic values for maximal stock.

### Extraction Price [^6]

We use the price of the different type of as :
![](price.jpg)

### Other data [^4]

We take the mean price of the different gas type according their proportion in the global gas production. 

To complete information about resource and demand the following data is also in the code :

|  Region  |gas type  |  current Reserve | Reserve unit |
| :------- | :--------:| ---------: | :-----------------: |
| World  | all type | 188100 | billion cubic metres |

### Sector using gas [^5]

| sector | proportion of the global demand per year in % | billion cubic metres|
| :------- | :--------:| :----------:|
|transport|7.3|279.0644|
|residential|29.9|1143.0172|
|commercial and public services	|12.9|493.1412|
|Industry|37|1414.436|
|other sector|12.9|493.1412|

Data implemented in the gas demand csv.

### References 

[^1]: Jon Claerbout and Francis Muir - "Hubbert math" (2020) - Retrieved from: 'http://sepwww.stanford.edu/sep/jon/hubbert.pdf'
[^2]: https://www.iea.org/data-and-statistics/data-product/world-energy-outlook-2021-free-dataset#tables-for-scenario-projections
[^3]: Wang, Jianliang & Bentley, Yongmei - "Modelling world natural gas production." (2020) Energy. Vol. 1363-1372. 10.1016/j.egyr.2020.05.018. - Retrieved from: 'https://www.researchgate.net/publication/341598129_Modelling_world_natural_gas_production'
[^4]: BP - Statistical Review of World Energy (2021) - Retrieved from: 'https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/energy-economics/statistical-review/bp-stats-review-2021-full-report.pdf'
[^5]: International Energy Agency - "World Energy Balances" (2020) - Retrieved from: 'https://www.iea.org/data-and-statistics/charts/world-coal-final-consumption-by-sector-2018'
[^6]: Roberto F. Aguilera - "Production costs of global conventional and unconventional petroleum" (2014) Energy Policy Volume 64 Pages 134-140 ISSN 0301-4215 - Retrieved from: 'https://doi.org/10.1016/j.enpol.2013.07.118.https://www.sciencedirect.com/science/article/abs/pii/S0301421513007763'
