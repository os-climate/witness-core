The Earth offers a finite amount of copper. This is a simplistic model of the copper consumption.

### Model principle

This model takes in input the annual demand of copper resource . It gives in output the copper price per year, the predicted production, the quantity of copper used each year in order to answer the copper demand, and the production in excess which is stocked for the future needs.
To calculate all this dataframe the model takes in input the demand.
The demand is simplistic : in constant growth so that the demand triple between 2020 and 2040.
When the reserve underground falls to an arbitrary number (500 Mt in this case), the extraction diminishes.
When there is too much difference between the extracted coppper and the demand (5 Mt here), the price rises.
Stock is yearly updated, it rises if the demand is lower than the extraction, and it diminishes if the extraction can not fulfill the demand.





### Copper data [^1]

|Year |Estimated World Resources [Mt]|World reserves [Mt] | Demand [Mt]|
| :------- | :---------- | :-----------: | :-----------: |
|2021 |3 500|880|26|




### Sector using gas [^2]

| sector | proportion of the global demand per year in % |
| :------- | :--------:|
|Building constructions|50|
|Infrastructure|22|
|Transportation	|5|
|Consummers durables|5|
|Commercial durables|10|
|Industrial durables|8|


### References

[^1]: US Geological Survey, Copper - "Hubbert math" (2020) - Retrieved from: 'https://pubs.usgs.gov/periodicals/mcs2022/mcs2022-copper.pdf'
[^2]: Branco W.Schipper, Hsiu-ChuanLin, Marco A.Meloni, KjellWansleeben, ReinoutHeijungs, Estervan der Voet - Estimating global copper demand until 2100 with regression and stock dynamics - Retrieved from : 'https://www.sciencedirect.com/science/article/pii/S0921344918300041'
