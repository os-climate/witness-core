The Earth offers a finite amount of oil. Theses oil stocks take different forms and according to the oil kind and the region it comes from, the price changes.
The first model just takes into account the world oil stock.

In order to find these parameters we need to find the data for the past production per year from the beginning of the production
The used data are taken from 1933 to 2020 there is several categories of oil which represent a percentage of the total production. These proportions change with the production year but in this model we take the repartition at year 2020.

production data sources: [^2] and [^3]

### Oil data

|Year |light 32%(Mt)|medium 54%(Mt)|heavy 13%(Mt)|unassigned(Mt)| production 1%(Mt)|
| :------- | :---------- | :-----------: | :---------: | :-----------------: |:-----------------:|
|1933|272|87|147|35|3|
|...|...|...|...|...|...|
|2020|1338|2258|544|42|

### Fitting [^4]

To fit the curve with the maximum reserve estimated by BP we adjust the beginning year of the regression in order to take the year start at the beginning of the current peak and get realistic values for maximal stock.

### Extraction Price [^5]

We use the price of the different types according to API type:
![](APII.png)

And we find the mean price:

![](API.jpg)


### Other data [^4]

The following data are integrated into the model

|  Region  |oil type  | current Reserve | Reserve unit |
| :------- | :--------:| ---------: | :-----------------: |
| World | crude_oil | 1732400 | million_barrels |

### Sector using oil [^6]

|  |proportion of the global demand per year in %|demand in million barrels|
|:------- | :--------:|:-----------------:|
|rail|0.8|283.536|
|aviation|8.3|2941.686|
|road|49.3|17472.906|
|navigation|6.8|2410.056|
|residential|5.4|1913.868|
|industry|7.2|2551.824|
|other sector|22.2|7868.124|
Data implemented in the oil demand csv.

### References

[^1]: Jon Claerbout and Francis Muir - "Hubbert math" (2020) - Retrieved from: 'http://sepwww.stanford.edu/sep/jon/hubbert.pdf'
[^2]: Eni S.p.A. - "World Oil Review 2020" - Retrieved from: 'https://www.eni.com/assets/documents/eng/scenari-energetici/WORLD-OIL-REVIEW-2020-vol1.pdf'
[^3]: Wikipedia - Retrieved from: 'https://fr.wikipedia.org/wiki/Pic_p%C3%A9trolier'
[^4]: BP - Statistical Review of World Energy (2021) - Retrieved from: 'https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/energy-economics/statistical-review/bp-stats-review-2021-full-report.pdf'
[^5]: IEA 2022; Oil Market Report - December 2019, https://www.iea.org/reports/oil-market-report-december-2019, License: CC BY 4.0.
[^6]: IEA 2022; World oil final consumption by sector, https://www.iea.org/data-and-statistics/charts/world-oil-final-consumption-by-sector-2018, License: CC BY 4.0.
