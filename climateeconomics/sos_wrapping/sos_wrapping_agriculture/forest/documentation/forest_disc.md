Forests are a natural consumer of $CO_2$. As $CO_2$ has a major impact on the temperature evolution, tracking the evolution of forests and their $CO_2$ consumption is important in order to get accurate results.
Many factors can lead to a reduction of the global forests surfaces, such as consequencies of temperature change and human activities. On the other side, environmental care and policies can lead to a rise of the global forest surface.

In this forest model, the global evolution of forest are impacted by 4 activities. There are afforestation and deforestation that respectively consists on planting and cutting trees. The two other activities are managed and unmanaged wood, which consist on planting trees, then harvest wood for different purpose (industrial or energy). These two last activities expand the forest surface but also generate biomass products.

## Model data

The forest model takes the following data as inputs:

- **year_start** is the first year of the study. Default is 2020.
- **year_end** is the final year of the study. Default is 2100.
- **time_step** represents the number of year between two data computation. Default is 1.
- **deforestation_investment** represents the money invested in the deforestation. Unit is G$.
- **deforestation_cost_per_ha** represents the average price to deforest 1ha of land. Unit is \$/ha. Default value is 12000 \$/ha [^1].
- **CO2_per_ha** represents the quantity of $CO_2$ captured by 1 hectare of forest during one year. Unit is $kgCO_2/ha/year$. Default value is 4000 kgC02/ha/year [^2].
As forest captures 16 Gt of $CO_2$ per year, reducing forest by 1% results in a deficit of $CO_2$ captured of 160 Mt. The value of 4000 kg$CO_2$/year/ha is coherent with these data.
- **Initial $CO_2$ emissions** represents $CO_2$ emissions balance of forest at the first year of the study. Unit is $GtCO_2$. Default value is -7.9 $GtCO_2$ at 2020, which is the value found at [^2].
- **reforestation_cost_per_ha** represents which is the average price to plant 1ha of tree. Unit is \$/ha. The default value is 13800\$/ha splitted as 10000\$/ha for land cost and 3800\$/ha to plant trees [^3].
- **reforestation_investment** represents the quantity of money dedicated to reforestation each year in billions of $.
- **wood techno dict** represents data use to compute price and production of biomass for managed wood and unmanaged wood.
- **managed wood initial prod** represents the production of biomass by managed wood at the first year of the study. Unit is TWh.
- **managed wood initial surface** represents the surface dedicated to managed wood, at the first year of the study. Unit is Gha. Default is 1.15Gha, at 2020.
- **managed wood invest before year start**represents the investments made into managed wood that impact the first years of the study. Unit is G$.
- **managed wood investment** represents the investment made into managed wood over the years. Unit is G$.
- **transport cost** represents the cost of transport of biomass in $/tons.
- **margin**, factor applied to the total cost of biomass to represent the commercial margin.
- **protected_forest** represents the surface of protected forest. Unit is Gha. Protected forest represents 21% of the global 4Hha forest surface, that is to say 0.84Gha.
- **unmanaged_forest_surface** represents the initial surface of forest that is not used for energy or industrial purpose. As 1.25Gha of forest are used, 0.84Gha are protected, and there are 4Gha of forest, the unused surface is 1.91Gha.
 
The outputs of the model are:

- **forest_surface_df** represents the major data of forest surface evolution over the year. Unit is Gha.
- **forest_surface_detail_df** represents detailed data about forest surface evolution over the year. Unit is Gha.
- **CO2_emitted_df** represents evolution of $CO_2$ captured by forest in $GtCO_2$.
- **CO2_emissions_detail_df** represents detailed data about $CO_2$ emitted by forest activities. Unit is $GtCO_2$.
- **CO2_land_emission_df** represents information about computed land emissions. Unit is $GtCO_2$.
- **managed_wood_df** represents data about managed wood prodution.
- **biomass_dry_detail_df** represents detailed data about biomass dry production.
- **biomass_dry_df** represents major data about biomass dry production.
- **techno_capital** represents represents the total capital allocated to the reforestation technology, in G$.
- **non_use_capital** represents is the unused capital of reforestation du to deforestation activities. Unit is G$.

## Global approach

The forest model has to track the global forest surface evolution, the wood harvested (more generally biomass) and $CO_2$ captured. To do this, the following assumptions are made.
The global forest surface is divided into 3 parts:
* Managed forest. These are the forests dedicated to long term biomass production thanks to management plans.
* Protected forest. These are the forests that are legally protected, and they will stay as they are. No management plan allowed.
* Unmanaged forest. These are forests that are now unused but they are not protected. As a result, they can be transformed by human activities.

Then, 3 different activities will impact these surfaces taken into account.
* Reforestation. This activities consists in planting trees, and thus increases the unmanaged forest surface, as the global forest surface.
* Deforestation. This activities cuts trees and reduce unmanaged forest surface as the global forest surface. Deforestation can not impact protected forest. Deforestation produces biomass as a one-time activities.
* Managed wood. This activities consists in managing forest to produce biomass regulary on a long term period. Investing in managed wood will increase managed forest surface and so the global forest surface.

Following paragraphs gives further details about each part of the model.


## Afforestation and reforestation

Deforestation and reforestation are activities that impact the evolution of the global forest surface.
They both impact unmanaged forests.

Deforestation is directly the **deforestation_surface** from the inputs.
Reforestation is calculated by
$$Reforestation\_surface = Reforestation\_investment / cost\_per\_ha$$

The cumulative value is the sum of all the forest surface evolution from the first year of the study to the given year of the data. Deforestation also produces biomass.

The surface deforested is removed from the existing forest surface. It firstly takes out unmanaged surfaces. When there is no more unmanaged trees to  down, managed forest are harvested. Then, when there is no more managed wood left, nothing is harvested, as protected forests can not be impacted by timber production.

## Managed wood

Managed wood defines all the forest under management plan in order to produce biomass on a long term period.

**Surface of forest**
Each year, a certain amount of money is invested into managed wood. This is an input data of the model. Knowing the price per ha (in **wood techno dict**) the surface added each year can be deduced by
$$Added\_surface = investment / price\_per\_ha$$
This price per ha takes into account planting tree, preparing ground, harvesting and other activities linked to wood management.
By adding the surface of forest planting each year, the cumulative surface is computed, which represent the total of managed or unmanaged wood added since the first year of the study.


**Biomass production**
The quantity of biomass produced by 1 ha is given by
$$biomasss\_per\_ha = quantity\_per\_ha \times average\_density / years\_between\_harvest / (1-recycle)$$
with:
quantity\_per\_ha : the average quantity of wood per ha in $m^3/ha$. This is the average between wood and residues.
average\_density : the average density of biomass in $kg/m^3$
years\_between\_harvest : the number of year between 2 harvesting in the same place
recycle : the percentage of biomass that comes from recycling

Knowing the surface of managed wood we can deduced the quantity of biomass produced.

**Biomass price**
Biomass is produced by managed forest and by deforestation. Each of these production way has its own price. As a result, the average price of biomass is the weighted average of managed wood and deforestation price.
$$biomass\_price = managed\_wood\_price \times managed\_wood\_part + deforestation\_price \times deforestation\_part$$
with 
$$deforestation\_part = \frac{deforestation\_production}{total\_biomass\_production}$$
and 
$$managed\_wood\_part = \frac{managed\_wood\_production}{total\_biomass\_production}$$


## $CO_2$ emissions
The land emissions can be computed with this formula:
$$CO2\_land\_emissions = Surface\_deforestation \times CO2\_per\_ha - Surface\_reforestation \times CO2\_per\_ha $$

The forest for energy emissions can be computed as following:
$$CO2\_emissions = CO2(by\_use\_biomass) - CO2(captured\_from\_growing\_trees) \ge 0$$

As a result, deforested surfaces does not capture $CO_2$ as they does not exist anymore.
Managed surfaces are neutral in $CO_2$ because $CO_2$ captured in wood is likely to be released when the wood is burn.

## Lost capital
In the forest model, there are 3 activities : deforestation, reforestation, managed wood, with their own investment. If reforestation and managed wood aimed at expand forest surface, deforestation diminishes it. As a result using money into opposite activities leads to a waste of money called lost capital.
As an example, investing in 10Mha of reforestation and 8Mha of deforestation results in a +2Mha of forest planted. Then, investment in reforestation is not optimized.
The following formula gives the detail of the lost capital computation.

**First case** : part of unmanaged wood is deforested
In this first case, trees that were planted in the past years are cut and the land will be used for another purpose than forestry.
$$reforestation\_lost\_capital = unmanaged\_deforested\_surface \times reforestation\_cost\_per\_ha$$

**Second case** : all unmanaged wood is deforested, part of managed wood is deforested
In this second case, all unmanaged wood is deforested and managed wood trees start to be cut. As in first case reforestation capital is lost. Moreover, some of managed wood capital is also lost.
$$reforestation\_lost\_capital = unmanaged\_deforested\_surface \times reforestation\_cost\_per\_ha$$
$$managed\_wood\_lost\_capital = managed\_deforested\_surface \times managed\_cost\_per\_ha$$

**Last case** : all that can be cut is cut
In this last case, all unmanaged wood is deforested, all managed wood is also deforested. Plus, money is over invested into deforestation. This exceed amount of money is lost because nothing can be cut.
$$reforestation\_lost\_capital = unmanaged\_deforested\_surface \times reforestation\_cost\_per\_ha$$
$$managed\_wood\_lost\_capital = managed\_deforested\_surface \times managed\_cost\_per\_ha$$
$$deforestation\_lost\_capital = exceed\_deforestation\_investment$$

## Model limitations
In this model, the quantity of $CO_2$ captured by ha of forest is assumed to be the same all over the world.  However, the $CO_2$ captured is influenced by the climate condition. Forest in tropical regions are more effective than forest in cold regions. As a result, cutting trees of the Amazon forest does not have the same impact than cutting trees in cold region, in term of captured $CO_2$.

## References

[^1]: LawnStarter, Pricing Guide: How much does it cost to clear land ?, found online at https://www.lawnstarter.com/blog/cost/clear-land-price/#:~:text=Expect%20to%20pay%20between%20%24733,higher%20your%20bill%20will%20be.
[^2]: World Resources Institute, Forests Absorb Twice As Much Carbon As They Emit Each Year, January 21, 2021 By Nancy Harris and David Gibbs, found online at https://www.wri.org/insights/forests-absorb-twice-much-carbon-they-emit-each-year
[^3]: Agriculture and Food Development Authority, Reforestation, found online at https://www.teagasc.ie/crops/forestry/advice/establishment/reforestation/
