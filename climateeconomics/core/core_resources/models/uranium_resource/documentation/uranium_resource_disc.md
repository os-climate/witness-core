The Earth offers a finite amount of uranium. These uranium stocks take different forms and according to the uranium kind and the region it come from, the price varies.
The first model takes into account the uranium kind extracted and the associated price.

In order to find these parameters we need to find the data for the past production per year from the beginning of the production
The used data are taken from 1950 to 2020 there is several categories of uranium which represent a percentage of the total production. These proportions change with the production year. In order to get a idea of this percentage we use a hypothesis that more expensive Uranium types will follow the curve shape of the past peak of uranium but with data which correspond to their different reserves proportion. We begin the Hubbert curve in the production data arround 2010 to get an idea of the maximum predicible production for each uranium types.

Production data sources [^2]

### Uranium data

The different types of uranium are identified according to their extraction price :

| Year | uranium 40 USD (tonnes) | uranium 80 USD (t) | uranium 130 USD (t) | uranium 260 USD (t) |
| :--- | :---------------------- | :----------------: | :-----------------: | :-----------------: |
| 1950 | 800                     |        500         |        2700         |        1000         |
| ...  | ...                     |        ...         |         ...         |         ...         |
| 2020 | 1338                    |        2258        |         544         |         42          |

### Fitting [^3]

To fit the curve with the maximum reserve estimate by BP we adjust the beginning year of the regression in order to take the year start at the beginning of the current peak and get realistic values for maximal stock.

### Other data [^3]

The following data are integrated into the model

| Region | uranium type | Price | Price unit | current Reserve | Reserve unit |
| :----- | :----------: | :---- | :--------: | --------------: | :----------: |
| World  | recoverable  | 40    |   USD/k    |          744500 |    tonnes    |
| world  | recoverable  | 80    |   USD/k    |         1243900 |    tonnes    |
| World  | recoverable  | 130   |   USD/k    |         3791700 |    tonnes    |
| World  | recoverable  | 260   |   USD/k    |         4723700 |    tonnes    |
| world  |   in situs   | 40    |   USD/k    |          882900 |    tonnes    |
| World  |   in situs   | 80    |   USD/k    |         1528100 |    tonnes    |
| World  |   in situs   | 130   |   USD/k    |         4971400 |    tonnes    |
| World  |   in situs   | 260   |   USD/k    |         6176700 |    tonnes    |

These data give the world identified stock. Next we calculate the reserve left each year and the price evolution associate. In this model we use all the cheaper resources before extracting the more expensive ones.
The resource price also depends on the world region we are living in. here the price correspond to the US market price. So, for more precision, we have to make a model base on the world region price.

### Sector using uranium

- Nuclear plant
- Medicine
- Nuclear weapon
- Colorant

The nuclear plant sector seems to be the one which uses the most uranium resource, the others using recycled uranium. Data on nuclear plant demand are easy to find, otherwise other sector demands are quite difficult to find.

Data associated are implemented in the csv Uranium demand.

### References

[^2]: Nuclear Energy Agency and International Atomic Energy Agency - "Uranium 2020: Resources, Production and Demand" - Retrieved from: '<https://www.oecd-nea.org/jcms/pl_52718/uranium-2020-resources-production-and-demand?details=true>'

[^3]: BP - Statistical Review of World Energy (2021) - Retrieved from: '<https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/energy-economics/statistical-review/bp-stats-review-2021-full-report.pdf>'
