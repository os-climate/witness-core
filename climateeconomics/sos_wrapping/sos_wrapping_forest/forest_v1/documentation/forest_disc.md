Forests are a natural consumer of CO2. As CO2 has a major impact on the temperature evolution, tracking the evolution of forests and their CO2 consumption is important in order to get precise results.
Many factors can lead to a reduction of the global forests surfaces, such as consequencies of temperature change and human activities. On the other side, environmental care and policies can lead to a rise of the global forest surface.

## Model data

The forest model takes the following data as inputs:

- **year_start**, the first year of the study. Default is 2020.
- **year_end**, the final year of the study. Default is 2100.
- **time_step**, the number of year between two data computation. Default is 1.
- **limit_deforestation_surface**, the maximum surface in Mha which can be deforested during the study.
- **deforestation_surface**, forest surface removed by year in Mha. Default is set to 10Mha per year (2020 value).
- **CO2_per_ha**, the quantity of CO2 captured by 1 hectare of forest during one year. Unit is kgCO2/ha/year. Default value is 4000kgC02/ha/year [^1].
As forest captures 16 Gt of CO2 per year, reducing forest by 1% results in a deficit of CO2 captured of 160 Mt. The value of 4000kgCO2/year/ha is coherent with these data.
- **Initial CO2 emissions**, CO2 emissions in GtCO2 due to deforestation at the first year of the study. Default value is 3.21 GtCO2 at 2020, which is the value found at [^2].
- **reforestation_cost_per_ha**, which is the average price to plant 1ha of tree. Unit is $/ha. The default value is 3800 $/ha [^3].
- **reforestation_investment**, the quantity of money dedicated to reforestation each year in billions of $.
 
The outputs of the model are:

- **forest_surface_df**, giving the evolution of forest surface year by year, and cumulative in Gha.
- **CO2_emitted_df**, gives evolution of CO2 captured by forest in GtCO2.

## Evolution of forest surface

Forest evolution is the sum of deforestation and reforestation contributions.
Deforestation is directly the **deforestation_surface** from the inputs.
Reforestation is calculated by
$$Reforestation\_surface = Reforestation\_investment / cost\_per\_ha$$

The cumulative value is the sum of all the forest surface evolution from the first year of the study to the given year of the data.

## Evolution of CO2 captured
The evolution of CO2 captured by forest is directly linked to the surface of forest. This evolution of CO2 captured is given by:
$$CO2\_captured\_evolution = Forest\_surface\_evolution * CO2\_per\_ha$$

## Model limitations
In this model, the quantity of CO2 captured by ha of forest is assumed to be the same all over the world.  However, the CO2 captured change with the climate condition. Forest in tropical regions are more effective than forest in cold regions. As a result, cutting trees of the Amazon forest does not have the same impact than cutting trees in cold region, in term of captured CO2.

## References

[^1]: World Resources Institute, Forests Absorb Twice As Much Carbon As They Emit Each Year, January 21, 2021 By Nancy Harris and David Gibbs, found online at https://www.wri.org/insights/forests-absorb-twice-much-carbon-they-emit-each-year
[^2]: Our World In Data, Global CO2 emissions from fossil fuels and land use change, found online at https://ourworldindata.org/co2-emissions
[^3]: Agriculture and Food Development Authority, Reforestation, found online at https://www.teagasc.ie/crops/forestry/advice/establishment/reforestation/
