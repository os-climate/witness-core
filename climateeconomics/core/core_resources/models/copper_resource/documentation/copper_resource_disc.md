The Earth offers a finite amount of copper.
The first model just takes into account the world copper stock.

In order to find these parameters we need to find the data for the past production per year from the beginning of the production
The used data are taken from 1925 to 2020.

production data sources: [^2]

### Copper data

| Year | Copper [Mt] |
| :--- | :---------- |
| 1925 | 1.761       |
| ...  | ...         |
| 2020 | 20.6        |

### Fitting [^3]

To fit the curve with the maximum reserve estimated by US Geological Survey we adjust the beginning year of the regression in order to take the year start at the beginning of the current peak and get realistic values for maximal stock.

### Extraction Price [^4]

We start with the price of 2020.
The price function is a simplistic function that depends on the ratio use_stock/demand.
If the demand is fulfilled, ratio = 1 and the price is at its minimum. If nothing can be provided to answer the demand, ratio = 0 and the price is at it's maximum.
In between, the price follows an affine curve :
$$price(ratio) = (price\_max - price\_min) (1 - ratio) + price\_min$$

### Other data [^3]

The following data are integrated into the model

| Region | material type | current Reserve |  Reserve unit  |
| :----- | :-----------: | --------------: | :------------: |
| World  |    copper     |            2100 | million_tonnes |

### Sector using copper [^5]

| Sector                              | proportion of the global demand per year in % | demand in Million tonnes |
| :---------------------------------- | :-------------------------------------------: | :----------------------: |
| Power generation                    |                     9.86                      |           2.47           |
| Power distribution and transmission |                     35.14                     |           8.78           |
| Construction                        |                      20                       |            5             |
| Appliance & electronics             |                     12.5                      |          3.125           |
| Transports                          |                     12.5                      |          3.125           |
| Other                               |                      10                       |           2.5            |

Data implemented in the copper input.

### References


[^2]: US Geological Survey - "Copper Statistics and Information" - Retrieved from: '<https://www.usgs.gov/centers/national-minerals-information-center/copper-statistics-and-information>'

[^3]: US Geological Survey - Mineral Commodity Summaries 2022 Copper - Retrieved from: '<https://pubs.usgs.gov/periodicals/mcs2022/mcs2022-copper.pdf>'

[^4]: Macrotrends - Copper Prices, 45 Year Historical Chart - Retrieved from : '<https://www.macrotrends.net/1476/copper-prices-historical-chart-data>'

[^5]: Copper Alliance - Copper Environmental Profile - Retrieved from : <https://copperalliance.org/sustainable-copper/about-copper/copper-environmental-profile/>
