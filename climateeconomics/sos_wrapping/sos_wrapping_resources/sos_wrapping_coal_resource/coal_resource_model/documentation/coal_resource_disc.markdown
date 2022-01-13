The Earth offers a finite amount of coal. Theses coal stocks take different forms and according to the coal kind and the region it comes from, the price changes.
The first model takes into account two coal families : 
* Anthracite and bituminous family is the most expensive stock and so it's the most fill stock (753639 million tons left)
* Lignite and sub-bituminous family is the less expensive stock and so the less fill stock (320469 million tons left) [^3]

### Model principle

This model takes in input the annual demand of coal resource from energy mix model (all resource demand dataframe). It gives in output the coal mean price per year, the predictable production per coal types, the quantity of coal used each year in order to answer the coal demand, and the production in excess which is stocked for the future needs.
To calculate all this dataframe the model takes in input the demand and the past production data of each coal types.
With these data we can calculate the Hubbert curves used in prediction of coal production.
The prediction on coal production shows when the addition of production and stocks will no longer be able to answer the annual demand.
Stocks are calculated with the production excess. Each year the model puts, in the stock, all the production not used in the year. 
The model uses as a priority the coal types which are the less expensive, then it stocks the more expensive one if there is production in excess.
The price is calculated as the mean value of all the extraction price. It takes each coal types price and calculate according to each coal proportion the resulting price.

### Hubbert model for production prediction [^1]

This model can give different informations: 
* The approximate year of a given resource production peak
* The estimate maximum quantity of the stock
* The projection on production capacity for the future year

The computing of this information is based on the extrapolation of the past production. With this extrapolation we find some parameters useful in the predictable production equation:
![](production.png)

In this equation we have:
- P: production per year
- Q: cumulative production per year from year start
- tau: is the year of coal peak
- w: is the frequency 
- Qinf: the maximum stock usable 

We can find w and Qinf with the linear regression between production (P) and cumulative production (Q).

![](regression.png)

In order to find these parameters we need to find the past production data per year from the beginning of the production
The used data are taken from 1950 to 2020 there are several categories of coal which represent a percentage of the total production. These proportions change with the production year but in this model we take the repartition at year 2020. 

production data sources : [^2] and [^3]

### Coal data

|Year |sub-bituminous and lignite (Mt)|anthracite and bituminous (Mt) |
| :------- | :---------- | :-----------: | 
|1950|540|1260|
|...|...|...|...|...|...|
|2020|2272.5|5302.5|

### Fitting [^3]

To fit the curve with the maximum reserve estimate by BP we adjust the beginning year of the regression in order to take the year start at the beginning of the current peak and get realistic values for maximal stock.

### Extraction Price [^4]

We use the mean price according to the proportion of each coal types. 

### Other data [^3]

The following identified stocks are integrated into the model :

|  Region  |Coal type  | Current reserve | Reserve unit |
| :------- | :--------:|  ---------: | :-----------------: |
| World  | anthracite and bituminous | 753639 | million tonnes |
| world  |  lignite and sub-bituminous | 320469 | million tonnes |

Next we calculate the reserve left each year and the price evolution associate. In this model we use all the cheaper resources before extracting the more expensive ones.
The resource price also depends on the world region we are living in. Here the price corresponds to the US market price. So, for more precision, we have to make a model based on the world region price.

### Sector using coal [^5]

|          |proportion of the global demand per year in % |demand in tonnes|
| :------- | :---------------------------------:| :--------------: |
|global|100|7921|
|service, fishing, agriculture|4.6|364.366|
|residential|7.5|594.075|
|Iron steel|32.5|2574.325|
|chemical, petrochemichal|9.7|768.337|
|Non metallic mineral|21.2|1679.252|
|Other sector|24.5|1940.645|

Data implemented in the coal demand csv.

### References 

[^1]: Jon Claerbout and Francis Muir - "Hubbert math" (2020) - Retrieved from: 'http://sepwww.stanford.edu/sep/jon/hubbert.pdf'
[^2]: Höök, M., Zittel, W., Schindler, J. & Aleklett, K. - "Global coal production outlooks based on a logistic model" (2010) Fuel, Vol. 89, Issue 11: 3546-3558 - Published online at 'http://dx.doi.org/10.1016/j.fuel.2010.06.013' - Retrieved from: 'https://www.diva-portal.org/smash/get/diva2:329110/FULLTEXT01.pdf'
[^3]: BP - Statistical Review of World Energy (2021) - Retrieved from: 'https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/energy-economics/statistical-review/bp-stats-review-2021-full-report.pdf'
[^4]: U.S. Energy Information Administration - "Coal explained" (2021) - Retrieved from: 'https://www.eia.gov/energyexplained/coal/prices-and-outlook.php'
[^5]: International Energy Agency - "World Energy Balances" (2020) - Retrieved from: 'https://www.iea.org/data-and-statistics/charts/world-coal-final-consumption-by-sector-2018'