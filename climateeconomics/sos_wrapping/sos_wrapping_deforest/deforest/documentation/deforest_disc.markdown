# Forest Model

Forests are a natural consumer of CO2. As CO2 has a major impact on the temperature evolution, tracking the evolution of forests and their CO2 consumption is important in order to get precise results.
Many factors can lead to a reduction of the global forests surfaces, such as consequencies of temperature change and human activities. On the other side, environmental care and policies can lead to a rise of the global forest surface.

## Model data

The forest model takes the following data as inputs:

- **year-start**, the first year of the study. Default is 2020.
- **year_end**, the final year of the study. Default is 2100.
- **time_step**, the number of year between two data computation. Default is 1.
- **forest_df**, dataframe with the global forest surface each year. Unit is giga-hectare (Gha).
- **forest_evolution_rate**, the percentage of forest surface removed each year. Unit is percentage.
- **CO2_per_ha**, the quantity of CO2 captured by 1 hectare of forest during one year. Unit is kgCO2/ha/year. Default value is 5000kgC02/ha/year [^1].

The outputs of the model are:
- **forest_surface_evol_df**, giving the evolution of forest surface year by year, and cumulative. Unit is Gha.
- **captured_CO2_evol_df**, gives evolution of CO2 captured by forest.

## Evolution of forest surface

The evolution of the forest surface is driven by the input **forest_evolution_rate**. If the value is negative, forest surface is reduced. If the value is positive, forest surface  rises. For a given year, the evolution of forest surface is :
$$ Forest\_surface\_evolution = Forest\_surface * forest\_evolution\_rate / 100$$
Results need to be divided by 100 because forest\_evolution\_rate is in percentage.
The cumulative value is the sum of all the forest surface evolution from the first year of the study to the given year of the data.

## Evolution of CO2 captured

The evolution of CO2 captured by forest is directly linked to the surface of forest. If forest are expending, more CO2 will be captured. If forests are reduced, less CO2 will be captured. This evolution of CO2 captured is f=given by:
$$ CO2\_captured\_evolution = Forest\_surface\_evolution * CO2\_per\_ha$$

## Model limitations

In this model, the quantity of CO2 captured by ha of forest is assumed to be the same all over the world.  However, the CO2 captured change with the climate condition. Forest in tropical regions are more effective than forest in cold regions.


## References

[^1]: Office National des Forets, https://www.onf.fr/vivre-la-foret/+/590::la-foret-et-le-bois-des-allies-pour-le-climat.html