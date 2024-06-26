The Earth offers a finite amount of platinum.
The first model just takes into account the world platinum stock.

In order to find these parameters we need to find the data for the past production per year from the beginning of the production
The used data are taken from 1925 to 2020.

production data sources: [^2]

### Platinum data

| Year | Platinum [Mt] |
| :--- | :------------ |
| 1967 | 4.07e-5       |
| ...  | ...           |
| 2020 | 1.66e-4       |

### Fitting [^1]

To fit the curve with the maximum reserve estimated by US Geological Survey we adjust the beginning year of the regression in order to take the year start at the beginning of the current peak and get realistic values for maximal stock.

### Extraction Price [^4]

We start with the price of 2020.
The price function is a simplistic function that depends on the ratio use_stock/demand.
If the demand is fulfilled, ratio = 1 and the price is at its minimum (2020's price). If nothing can be provided to answer the demand, ratio = 0 and the price is at it's maximum (5 times 2020's price).
In between, the price follows an affine curve :
$$price(ratio) = (price\_max - price\_min) (1 - ratio) + price\_min$$

### Other data [^3]

The following data are integrated into the model

| Region | material type | current Reserve |  Reserve unit  |
| :----- | :-----------: | --------------: | :------------: |
| World  |   platinum    |          0.0354 | million_tonnes |

### Sector using platinum in 2020 [^5]

| Sector                    | proportion of the global demand per year in % | demand in Million tonnes |
| :------------------------ | :-------------------------------------------: | :----------------------: |
| Exhaust treatment systems |                      36                       |          8.1e-5          |
| Jewelry                   |                      26                       |          5.8e-5          |
| Chemical catalysts        |                       9                       |          2.0e-5          |
| Glass production          |                       8                       |          1.8e-5          |
| Electronics               |                       3                       |          7.0e-6          |
| Other                     |                      18                       |          4.0e-5          |

Data implemented in the platinum input.

### References

[^1]: Jon Claerbout and Francis Muir - "Hubbert math" (2020) - Retrieved from: '<http://sepwww.stanford.edu/sep/jon/hubbert.pdf>'

[^2]: US Geological Survey - "Platinum-Group Metals Statistics and Information" - Retrieved from: '<https://www.usgs.gov/centers/national-minerals-information-center/platinum-group-metals-statistics-and-information>'

[^3]: Fiche Critique Platine 2015 - Le platine (Pt) – éléments de criticité - Retrieved from: '<https://www.mineralinfo.fr/sites/default/files/documents/2020-12/fichecriticiteplatine-publique150409.pdf>'

[^4]: Statista - Average global platinum closing price from 2016 to 2022 - Retrieved from : '<https://www.statista.com/statistics/254519/average-platinum-price/>'

[^5]: Statista - Consumption of platinum worldwide in 2021, by industry - Retrieved from : '<https://www.statista.com/statistics/693866/platinum-consumption-worldwide-by-industry/#:~:text=In%202021%2C%20exhaust%20treatment%20systems,consumed%20worldwide%20in%20that%20year>'
