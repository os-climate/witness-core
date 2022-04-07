Forests are a natural consumer of CO2. As CO2 has a major impact on the temperature evolution, tracking the evolution of forests and their CO2 consumption is important in order to get precise results.
Many factors can lead to a reduction of the global forests surfaces, such as consequencies of temperature change and human activities. On the other side, environmental care and policies can lead to a rise of the global forest surface.

In this forest model, the global evolution of forest are impacted by 4 activities. There are afforestation and deforestation that respectively consists on planting and cutting trees. The two other activities are managed and unmanaged wood, which consist on planting trees, then harvest wood for different purpose (industrial or energy). these two last activities expand the forest surface but also generate biomass products.

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
- **wood tehcno dict**, data use to compute price and production of biomass for managed wood and unmanaged wood.
- **managed wood initial prod**, which is the production of biomass by managed wood at the first year of the study. Unit is TWh.
- **managed wood initial surface**, which is the surface dedicated to managed wood, at the first year of the study. Unit is Gha. Default is 1.15Gha, at 2020.
- **managed wood invest before year start**, which are the investments made into managed wood that impact the first years of the study. Unit is G$.
- **managed wood investment**, which are the investment made into managed wood over the years. Unit is G$.
- **unmanaged wood initial prod**, which is the production of biomass by unmanaged wood at the first year of the study. Unit is TWh.
- **unmanaged wood initial surface**, which is the surface dedicated to unmanaged wood, at the first year of the study. Unit is Gha. Default is 1.15Gha, at 2020.
- **unmanaged wood invest before year start**, which are the investments made into unmanaged wood that impact the first years of the study. Unit is G$.
- **unmanaged wood investment**, which are the investment made into unmanaged wood over the years. Unit is G$.
- **transport cost**, which is the cost of transport of biomass. Unit is $/ton.
- **margin**, factor applied to the total cost of biomass to represent the commercial margin.
- **unused_forest_surface**, the initial surface of forest that is not used for energy or industrial purpose. As 1.25Gha of forest are used and there are 4Gha of forest, the unused surface is 2.75Gha.
 
The outputs of the model are:

- **forest_surface_df**, giving the major data of forest surface evolution over the year. Unit is Gha.
- **forest_surface_detail_df**, giving detailed data about forest surface evolution over the year. Unit is Gha.
- **CO2_emitted_df**, gives evolution of CO2 captured by forest in GtCO2.
- **CO2_emissions_detail_df**, gives detailed data about CO2 emitted by forest activities. Unit is GtCO2.
- **managed_wood_df**, gives data about managed wood prodution.
- **unmanaged_wood_df**, gives data about unmanaged wood production.
- **biomass_dry_detail_df**, gives detailed data about biomass dry production.
- **biomass_dry_df**, gives major data about biomass dry production.
- **techno_capital**, which represents the total capital allocated to the reforestation technology, in G$.
- **non_use_capital**, which is the unused capital of reforestation du to deforestation activities. Unit is G$.

## Afforestation and deforestation

Forest evolution is the sum of deforestation and reforestation contributions.
Deforestation is directly the **deforestation_surface** from the inputs.
Reforestation is calculated by
$$Reforestation\_surface = Reforestation\_investment / cost\_per\_ha$$

The cumulative value is the sum of all the forest surface evolution from the first year of the study to the given year of the data.

## Managed wood and unmanaged wood

Managed wood and unmanaged wood are two way to plant trees and then to harvest biomass such as wood or residues. Managed wood concerns forest that are under management plan and unmanaged wood does not have management plan. This represent in 2020 around of 1.25 Gha, with 92% are managed and 8% non managed.

**Surface of forest**
Each year, a certain amount of money is invested into managed wood or unmanaged wood. This is an input data of the model. Knowing the price per ha (in **wood tehcno dict**) the surface added each year can be deduced by
$$Added\_surface = investment / price\_per\_ha$$
This price per ha take into account planting tree, preparing ground, harvesting and other activities linked to wood management.
By adding the surface of forest planting each year, the cumulative surface is computed, which represent the total of managed or unmanaged wood added since the first year of the study.

**Biomass production**
The quantity of biomass produced by 1 ha is given by
$$biomasss\_per\_ha = quantity\_per\_ha * average\_density / years\_between\_harvest / (1-recycle)$$
with:
quantity\_per\_ha : the average quantity of wood per ha in m^3/ha. This is the average between wood and residues.
average\_density : the average density of biomass in kg/m^3
years\_between\_harvest : the number of year between 2 harvesting in the same place
recycle : the percentage of biomass that comes from recycling

Knowing the surface of managed or unmanaged wood we can deduced the quantity of biomass produced.

**Biomass price**
For managed wood and unmanaged wood, the cost per ha is spread over the year using crf. Then the cost of transport is added and margin is applied.

The biomass price is the weighted average of managed  and unmanaged biomass price, according to their production quantity.


## Evolution of CO2 captured
The evolution of CO2 captured by forest is directly linked to the surface of forest. This evolution of CO2 captured is given by:
$$CO2\_captured\_evolution = Forest\_surface\_evolution * CO2\_per\_ha$$

## Lost capital
In order to do not waste money, the model will compute the value called lost capital that will be given to the appropriate model. This concerns reforestation and deforestation activities. As they are opposite activities, it is a waste of money and capital to invest in deforestation and reforestation.
For example, adding 10Mha of forest and removing 12Mha of forest results in removing 2Mha of forest. But the capital for adding the forest is lost.
As a formula, we use:
$$lost_capital = min(reforested\_surface, deforrested\_surface) * cost\_per\_ha$$

## Model limitations
In this model, the quantity of CO2 captured by ha of forest is assumed to be the same all over the world.  However, the CO2 captured change with the climate condition. Forest in tropical regions are more effective than forest in cold regions. As a result, cutting trees of the Amazon forest does not have the same impact than cutting trees in cold region, in term of captured CO2.

## References

[^1]: World Resources Institute, Forests Absorb Twice As Much Carbon As They Emit Each Year, January 21, 2021 By Nancy Harris and David Gibbs, found online at https://www.wri.org/insights/forests-absorb-twice-much-carbon-they-emit-each-year
[^2]: Our World In Data, Global CO2 emissions from fossil fuels and land use change, found online at https://ourworldindata.org/co2-emissions
[^3]: Agriculture and Food Development Authority, Reforestation, found online at https://www.teagasc.ie/crops/forestry/advice/establishment/reforestation/
