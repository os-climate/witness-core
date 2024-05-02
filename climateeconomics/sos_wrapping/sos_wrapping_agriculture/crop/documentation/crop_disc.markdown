## Agricultural lands

Agricultural lands are divided into 2 types of fields that are mixed in this model.
Crops for cereal, vegetables or other production, and grazing lands for switchgrass or other products.

The area available for crops is 1.1Gha and for grazing lands 3.8 Gha for a total around 4.9Gha[^1].

![](total_crop_lands.png)
(source: ourworldindata[^1])

## Agricultural products

This model computes the biomass dry produced with crops for energy.
The used initial production is 3333 TWh[^2]. We consider in the article conventional crop for energy and short-rotation woody crops are crops.
The harvest in the model is done at the end of each year. The farms have a lifetime of 50 years.

**crops**
The residues are not included in crop but in organic waste in another stream.
As most of the crops are for the food industry (human or animal), some of it is used for the energy sector, 
it is called energy crop. Only 0.1% of the total biomass production are from energy crops[^5].

The crop production is computed from an average crop yield at 2903kg/ha with 4070kg/ha for 1.1Gha of crop 
fields[^6] and an average crop yield of 2565.68kg/ha for 3.8Gha of grazing lands.
The total yield (crop + 25% residue) is:
$$Y = 2903kg/ha * 1.25 = 3628.75kg/ha$$

The crop energy production computed corresponds to a mix of conventional crops for energy and short-rotation woody crops. 
The crop production dedicated to industry (food or other) has been removed.

## Agricultural sectors

Crops model computes the land surface needed to feed humanity and how much crops can be used for energy.

**Land for food**

Food is one of the primary needs of human beings. 
However, it is also a real challenge to produce food with the limited land space available on our planet, 
and the large population.
The agriculture model aims at answering the following question:
Given a food diet for humanity, how much space is needed to feed everyone ?
This model allows to test different food diets, and computes the agriculture land area  
needed to produce enough food to feed all human beings.

**crops for energy**

crops for energy must not be in competition with feeding the humanity. 
The production of biomass dry by crops for energy is computed with a given investment and the remaining surface. 


## Model data
This section lists the inputs and outputs of the model.

The model inputs are:
* **year start**, this defines the starting year of the study.
* **year end**, this defines the last year of the study.
* **population_df**, this gives the population number, for each year of the study. The unit is in millions of people.
* **temperature_df**, gives the temperature in degree Celsius above pre-industrial level for each year of the study. 
* **starting diet**, this gives the food diet for the first year of the study. For each food category of the diet, 
the quantity in kg/person/year needs to be filled out. The different foods taken into account are:
red meat, white meat, milk, eggs, rice and maize, cereals, fruits and vegetables, fish and other including all 
the remaining foods,  such as sugar, all kind of nuts and oils, cocoa, coffee, tea, alcohol, soybeans, honey. 
The default diet per capita per year in kg of food is:

|Red meat| White meat |Milk|Eggs|Rice and Maize| cereals |Fruit and vegetables|Fish|          Other | Unit            |
| - |:----------:|-|-|:------------:|:-------:|:------------------:|---:|---------------:|-----------------|
|13.43|   31.02|73.07|10.45|       98.06       |  10.3 |266.28 |23.38|         177.02 |  kg/person/year |

These data are the average world diet [^7], in other words it represents the world food produced (not eaten) normalized by the 
world population. This leads to a daily diet of 2927 kcal/person. It therefore includes the food wasted, namely not eaten.
Since about 30% of the food produced is wasted [^22][^26], the daily diet actually eaten is around 2049 kcal/person. 

* **m2 per kg**, gives the average area needed to produce 1kg of the considered food. The unit is m^2/kg. By default, data comes from [^8]. 
The value for category other is adjusted so that the total land use of 4.9 Gha is reached considering the average diet for a population of 
7.8billions of people [^10]. 
At this stage, water area used for farm fishing is not accounted for in the crop model and is therefore set to 0.

* **kcal per kg**, gives the number of kilocalories for 1kg of the considered food. The unit is kcal/kg.
Extracted from [^9], the default values are:

|Red meat|White meat|Milk|Eggs|Rice and Maize| cereals |Fruit and vegetables|    Fish |  Other | Unit|
| --- |:--------:|----|----|:------------:|:-------:|:------------------:|--------:|-------:|---------|
|1551|2131|921|1425|2572|   2937   |543|     609 |   2582 | kcal/kg |


* **red meat calories per day**, gives the calories(kcal) of red meat in a person's diet per day.
* **white meat calory percentage**, gives the calories(kcal) of white meat in a person's diet per day.
* **fish calories per day**, gives the calories(kcal) of fish in a person's diet per day.
* **vegetable and carbs calories per day**, gives the calories(kcal) of vegetables and fruits in a person's diet per day.
* **egg and milk calories per day**, gives the calories(kcal) of egg and milk in a person's diet per day.
* **other calories per day**, gives the calories(kcal) in a person's diet per day of the food that does not belong to the categories above.

- **CO2_from_production**: Represent the CO2 emitted while the biomass dry production. CO2 is absorbed by the plants in order to be zero emissions in the end. Emissions from tractors are taken into account in the raw to net energy factor,
- **CO2_from_production_unit**: Unit of the CO2_from_production value. In kgCO2/kgBiomassDry by default,
- **Capex_init**: CAPEX of crop exploitation. See details on costs,
- **Opex_percentage**: OPEX of crop exploitation. See details on costs,
- **lifetime**: lifetime of an agriculture activities, initilized to 50 years,
- **residue_density_percentage**: percentage of residue, at 0.25 by default,
- **density_per_ha**: density of crop product by hectare, at 3628.75 kg/ha by default,
- **density_per_ha_unit**:unit of the density of crop, at kg/ha by default,
- **crop_percentage_for_energy**: crop percentage for energy, at 0.001 by default
- **residue_percentage_for_energy**: residue percentage for energy, at 0.1 by default
- **crop_residue_price_percent_dif**: ratio between the average price of residue price (23$/t) on average crop price (60$/t), 38% by default.
- **construction_delay**: laps of time before the first harvest, at 1 year by default
- **crop_investment**: investment for crops for energy
- **initial_production**: crop production for energy in 2020

The model outputs are:

* **agriculture productivity evolution**, gives the evolution of agriculture productivity overtime, due to climat change. Unit is %.
* **food land surface**, gives the surface needed to produce the amount of food of the diet. The unit is Gha.
* **food land surface percentage**, gives the share of the surface of each type of food, in %.
* **final diet**, gives the diet evolution over the years, in kg/person/year, for each type of food.
* **Total food land surface**, gives the total amount of land needed for food, in Gha.
* **CO2 Emissions**: CO2 emissions from crop production and residues for energy sector
* **CO2 land Emissions**:  CO2 land emissions from food production and crop energy production that will be transferred to the carbon emission model
* **CH4 land Emissions**:  COH4 land emissions from food production and crop energy production that will be transferred to the carbon emission model
* **Required land surface**: Surface of crop energy and food land
* **biomass dry production**: crop production for energy sector with residues for energy sector
* **biomass dry prices**: detailed of prices
* **Detailed mix prices**: detailed average price of crop and residues

## From a food diet to a surface
This section aims to explain how the model goes from a food diet to a surface to produce the required food.
The following picture shows the "how to" process.
![](diet_to_surface.PNG)

First, given the population and the diet per person, the quantity of food is deduced:
$$quantity\_of\_food = population * diet\_per\_person$$

Then, knowing the conversion factor between mass of food and surface needed to produce it, the required surface is given by:
$$required\_surface = quantity\_of\_food * square\_meter\_per\_food$$

To all of this, an other contribution will be added to the agriculture surface. This other contribution represents the use of agriculture for other purpose than producing the 7 food types already taken into account. It mainly takes into account :
cocoa - coffee - olive - sugar - oil(palm, sunflower,...) - tea - grapes(wine) - tobacco - yams - natural rubber - millet - textile fiber (cotton and other)...

## Changing the diet
This section aims at explaining how the diet change works in the agriculture model.

In this model, the user is able to change diet over time, then, to observe the effect of the changes.
First of all, when a diet changes, one of the risk is to be undernourished. 
To prevent that, the diet will remain at the same amount of kcal all the time.
When the term "convert" a food X to an other Y is used, it means suppress a quantity of food X, which corresponds to an amount of E kcal, 
and then add an amount of food Y that fill the E kcal removed.

The model starts from the base-diet given in input, **diet_df**.
The first change is to convert red meat and white meat, using **red_meat_percentage** and **white_meat_percentage** 
to vegetables (fruit and vegetables, potatoes and rice and maize).
This will give the percentage converted, based on the base-diet.
For red_meat_percentage = 6.86%, nothing will change, it is the same as the starting diet.
For red_meat_percentage = 0%, all red meat is removed, and missing kcal are filled by additional vegetables.
The red meat percentage must be between 10% and 1%.
The white meat percentage must be between 20% and 5%.
Eggs and milk are not impacted.

The following picture shows the different steps.
![](diet_update.PNG)

First, the new quantity of red meat is calculated:
$$base\_diet\_red\_meat = total\_kcal * red\_meat\_percentage / 100 / kg\_to\_kcal\_red\_meat$$

This red\_meat\_removed corresponds to a quantity of kcal determined by:
$$red\_energy\_removed = base\_diet\_red\_meat - total\_kcal * red\_meat\_percentage / 100$$

Then, the new quantity of white meat is calculated:
$$base\_diet\_white\_meat = total\_kcal * white\_meat\_percentage / 100 / kg\_to\_kcal\_white\_meat$$

This white\_meat\_removed corresponds to a quantity of kcal determined by:
$$white\_energy\_removed = base\_diet\_white\_meat - total\_kcal * white\_meat\_percentage / 100$$

The calories consumed by day are given for red meat and white meat. Fruits, vegetables, potatoes, rice and maize are given 
by the variable "vegetables and carbs". Calories per day of eggs and milk are given by another variable.
The base-diet is used to compute the actual proportion of fruits and vegetables, potatoes, rice and maize to distribute
calories from "vegetables and carbs" variable. It is also used to compute the actual proportion of eggs and milk.


## Climate change impact
The increase in temperature due to global warming has consequences on crop productivity. 
Effects of global warming on agriculture are for example drought, flooding and increased crops water needs.   
Some models estimate the impact of increased temperature on crop productivity but results are quite disparate.
We therefore chose to implement one impact for all agriculture productivity and not one per type of crop 
(maize, rice, wheat...). Our bilbiography to define the effect includes IPCC[^11] 
(Intergovernmental Panel on Climate Change), the FAO[^12] (Food and Agriculture Organization), 
Carl-Friedrich Schleussner et al (2018)[^13] and Rosenzweig et al. (2014)[^14].   
The surface required taking into account climate change is:
$$Required\_surface\_with\_global\_warming = required\_surface * (1- productivity\_change)$$
$$Productivity\_change = \alpha T^2+bT$$ 
$$with~~T_t = temperature_t -temperature_{t=0}$$ 
By default $\alpha$ and $b$ are respectively set at  -0.00833 and - 0.04167. So that at +2 degrees (above pre industrial level) the decrease in productivity is 5% and 30% at +5 degrees celsius.   
Then, when temperature rises, the surface required to produce the same amount of food increases by the share of production reduction. 

Greenhouse Gas emissions of food production and crops are computed from ratios of kgCO2eq per kg:
$$GHG(food) = kgCO2eq\_ghg(food) \times land\_use(food)$$ 
The kgCO2eq_ghg coefficient for each food is partially given in [^18] (see graph below). They are however only 
considered accross livestock and crops, emissions for land use and retail being accounted for elsewhere.
![](food-emissions-supply-chain.png)
However, [^18] shows more categories of food than the 7 mentioned above. They therefore need to be grouped as shown in the table below. 

|Red meat|White meat|Milk|Eggs|Rice and Maize| cereals |Fruit and vegetables|    Fish |                                                   Other |
| --- |:--------:|----|----|:------------:|:-------:|:------------------:|--------:|--------------------------------------------------------:|
|Beef herd, dairy herd, Lamb & Mutton|Pig, poultry|Milk|Eggs|Rice, Maize|   Barely, Oatmeal, Wheat and Rye   |Apples, Bananas, Berries and grapes, Brassicas, Cassava, Cirtus fruit, Onions and leeks, other fruit, other pulses, other vegetables, peas, potatoes, root vegetables, tomatoes|     Fish(farmed) |   Sugar, cheese, coffee, chocolate, oils, soybean, wine |

For each food sub-category, the kgCO2eq_ghg coefficient is weighted by the world's production of that sub-category taken 
from FAO's values of food production in 2019 [^20].
NB: FAO does not distinguish production for dairy and beef herd nor between wild and farm fish. The percentage of meat 
and fish coming respectively from dairy beef and farm fish are considered to be respectively 80% [^23] and 53% [^24].
For the other sub-categories, FAO actually introduces sub-sub-categories which production is summed up to obtain 
the sub-category worlwide production. The actual data pre-processing approach is extensively described in an internal 
Capgemini document [^25].

The raw kgCO2eq coefficients obtained above applied to the average diet surprinsingly overestimate the ghg for livestock and crops 
by a factor of 1.92. Therefore, the kgCO2eq_ghg coefficients are scaled down by this factor in order to obtain the 
expected global ghg emissions of 7.89 GtCO2eq for livestock and crops (see OurWorldInData global Food Grenhouse gas 
emissions [^17]). Again, we consider only emissions from Crops and Livestock as all supply chain and land use emissions are computed elsewhere. 
![](How-much-of-GHGs-come-from-food.png)

N2O emissions are estimated from global crop and pastures emissions of respectively 0.28 and 0.86 GtCO2eq (considering 
a ghg power of 28). Values are computed from the 2019 FAO stats [^21] considering that the N2O emissions for:
* crop come from crop residues, burning crop residues, manure management, 
manure applied on soils, Synthetic fertilizers, fertilizers and pesticides manufacturing
* pastures come from manure left on pastures and drained organic soils

It is assumed that cereals, fruitveggies and maize&rice and other have N2O emissions from crops, the rest from pastures. 
Then the kgCO2eq_N2O for N2O for each food category is computed from:
$$kgCO2eq\_N2O = emission(N2O) \times land\_use(food)/Total\_land\_use$$

CH4 emissions are initially obtained from the CH4 emission ratio provided by OurWorldInData, 
Carbon footprint and methane [^19], which allows to identify the CH4 part in these global GHG emissions.
However, those values are valid accross the supply chain whereas we are restricting the emissions computation 
accross livestock and crops only, excluding contributions of emissions from land use and retail. 
Therefore, those ratio need be corrected. In particular, for red meat, methane emissions are mostly due to enteric 
fermentation and manure management. Those steps do not occur during food pre and post-process, then CH4/ghg of [^19] 
is underestimated accross livestock and crop. 
* Global CH4 emissions for livestock 
(=red meat, white meat, eggs and milk) account for 3.08 GtCO2eq [^21]. NB: this figure is obtained while 
discarding the agri systems waste disposal that is considered as part of the post-production process [^25] (p.2).
It is therefore not considered for livestock & crops CH4 emissions accounting. 
* Global CH4 emissions for crop account for 0.7 Gt CO2eq, mostly corresponding to rice cultivation, which is 
already well estimated from the CH4/ghg ratio. 

To recalibrate the kgCO2eq_CH4 coefficient for livestock, tt is assumed that those coefficients are identical 
for each livestock category for manure management. The remaining CH4 emissions from enteric fermentation must then 
be split between red meat and milk (due to ruminants). This accounts for 2.81 GtCO2eq. 
In practice, those 2.81 GtCO2eq are split between red meat and milk following the ratio of CH4 obtained between 
those 2 categories from the CH4 emission ratio. For those 2 categories, this however leads to CH4 emissions values (CO2eq) 
higher than the ghg values computed. Then, for the red meat and milk categories, the CH4 value taken = ghg - N2O emissions 
(which are assumed to be correct) leading to CO2 emissions of 0 for red meat and milk. 
This approach is debatable, but it's the only way to recompute the global CH4 emissions with 8% error 
while keeping accurate the global ghg and N2O emissions.  

## Food waste
Once the food is produced, part of it (30% [^22][^26])is not send to the population model to represent the waste that exists between production and consumption.


## Results
This sections aims at describing the results charts.

* Surface taken to produce food over time:
This is a bar graph, displaying the surface needed (Gha) over the time (years of the study).
The detail is given for each food considered. Plus, a line plots the total surface needed to feed humanity, with the considered diet.
* Share of the surface used to produce food over time:
This graph display the same information than the last one, but values are normalized in percentage.
* evolution of the diet overtime:
This graphs gives the diet evolution for each year of the study, in kg / person / year, for each food.

## Model limitation
This section aims at giving the limitations of the model.

The agriculture model gives output data as the result of different computations described previously. 
However, the different hypothesis lead to limitations that should be kept in mind in order to have a critical 
point of view regarding the results.
The following lists the limitations of the agriculture model:

* The model does not allow a modification of the global kcal amount of the diet. This kcal amount is set by the initial diet, and will remain the same.
* The model does not test the feasibility of the diet change. For example, convert 100% of red meat or white meat 
in one year is allowed in the model, even if it is probably not realisable.
* The model considers a global average diet, and does not include change depending on country or habits.
* The land use to produce food are assumed to be able to produce any type of food, 
with no regards of average weather or climate conditions. 
For example equatorial or temperate climate may have different affinity with food production.
* The CH4 emissions for livestock are underestimated to keep global ghg estimates accurate
* Food waste modeling: Food can be wasted during two stages [^22][^26] (harvest to retail - 17%, retail to consumption - 13%). Currently, these two stages are not represented, and all the wasted food comes from the crop model. Later, harvest to retail wastes should be included in the crop model, and retail to consumption wastes should be included in the population model.
  


## Biomass dry Production

Initial production: production of energy crop in 2020.

Initial_production=4.9 Gha of agricultural lands * density_per_ha * 3.36 kWh/kg calorific value * crop_percentage_for_energy

The model computes the crop energy production and then add the residue production part for energy from the surface for food.

$$Residue\_production\_for\_energy=total\_production * residue\_density\_percentage +  \\ residue\_energy\_production\_from\_food$$

$$Crop\_production\_for\_energy=crop\_density\_percentage*total\_production$$

Then :

$$total\_production\_for\_energy=Residue\_production\_for\_energy+Crop\_production\_for\_energy$$

## Land use

The computed land-use amount of hectares is the agricultural area for energy crops (only the crop part) with the following computation:

$$NumberOfHa=\frac{CropProductionForEnergy}{density\_per\_ha*calorific\_value}$$

With: 
- CropProductionForEnergy, the production of crop and residue for energy sector computed by this model

## Costs

For CAPEX computation:
 - crop production: 237.95 €/ha (717$/acre)[^15]

For OPEX computation:
  - crop harvest and processing: 87.74 €/ha (264.4$/acre)[^15]
  - residue harvest (22$/t) + fertilizing (23$/t): 37.54 €/ha[^16]
  
The computed price is the mixed price of crop and residue. Details in the composition of prices of crop and residue is shown in the graphics named "Detailed Price of energy crop technology over the years". 
Prices are computed with the input parameter crop_residue_price_percent_dif.

## References

[^1]: OurWorldInData, land use over the long term, https://ourworldindata.org/land-use#agricultural-land-use-over-the-long-run
[^2]: IEA, What does net-zero emissions by 2050 mean for bioenergy and land use?, https://www.iea.org/articles/what-does-net-zero-emissions-by-2050-mean-for-bioenergy-and-land-use
[^3]: World Bioenergy Association, Global Energy Statistics 2019, http://www.worldbioenergy.org/uploads/191129%20WBA%20GBS%202019_HQ.pdf
[^4]:  World Bioenergy Association, Global biomass potential towards 2035, http://www.worldbioenergy.org/uploads/Factsheet_Biomass%20potential.pdf
[^5]: Bioenergy Europe, Biomass for energy: agricultural residues and energy crops, https://bioenergyeurope.org/component/attachments/attachments.html?id=561&task=download
[^6]: The world bank, Cereal yield kg per hectare, https://data.worldbank.org/indicator/AG.YLD.CREL.KG
[^7]: FAO, FAOSTAT, Food Balances (2014-),  Latest update: April 14 2021, Accessed August 2021,  http://www.fao.org/faostat/en/#data/FBS
[^8]: Poore, J., & Nemecek, T. (2018). Reducing food's environmental impacts through producers and consumers. Science, 360(6392), 987-992. Published online at OurWorldInData.org. Retrieved from: https://ourworldindata.org/grapher/land-use-per-kg-poore
[^9]: FAO, Food Balance Sheets A handbook, Annex 1 : Food Composition Tables, http://www.fao.org/3/X9892E/X9892e05.htm#P8217_125315
[^10]: Knoema, Production Statistics - Crops, Crops Processed, https://knoema.com/FAOPRDSC2020/production-statistics-crops-crops-processed?country=1002250-world
[^11]: IPCC, 2007. Climate Change 2007: Impacts, Adaptation and Vulnerability Contribution of Working Group II to the Fourth Assessment. Report of the Intergovernmental Panel on Climate Change, M.L. Parry, O.F. Canziani, J.P. Palutikof, P.J. van der Linden and C.E. Hanson, Eds., Cambridge University Press, Cambridge, UK, 976pp.
[^12]: FAO, 2019. The State of Food and Agriculture 2016 (SOFA): Climate Change, Agriculture and Food Security.
[^13]: Rosenzweig, C., Elliott, J., Deryng, D., Ruane, A.C., Müller, C., Arneth, A., Boote, K.J., Folberth, C., Glotter, M., Khabarov, N. and Neumann, K., 2014. Assessing agricultural risks of climate change in the 21st century in a global gridded crop model intercomparison. Proceedings of the national academy of sciences, 111(9), pp.3268-3273.
[^14]: Schleussner, C.F., Deryng, D., Muller, C., Elliott, J., Saeed, F., Folberth, C., Liu, W., Wang, X., Pugh, T.A., Thiery, W. and Seneviratne, S.I., 2018. Crop productivity changes in 1.5 C and 2 C worlds under climate sensitivity uncertainty. Environmental Research Letters, 13(6), p.064007.
[^15]: Manitoba, Crops production costs - 2021, gov.mb.ca/agriculture/farm-management/production-economics/pubs/cop-crop-production.pdf
[^16]: United States Department of Agriculture, 2016, Harvesting Crop Residue: What’s it worth?, https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcseprd1298023.pdf
[^17]: OurWorldInData: Food Grenhouse gas emissions, https://ourworldindata.org/food-ghg-emissions
[^18]: OurWorldInData: Environmental impacts of food, https://ourworldindata.org/environmental-impacts-of-food#co2-and-greenhouse-gas-emissions
[^19]: OurWorldInData: Carbon footprint and methane, https://ourworldindata.org/carbon-footprint-food-methane
[^20]: Food and Agriculture Organisation, World Statistics, https://www.fao.org/faostat/en/#data/FBS
[^21]: Food and Agriculture Organisation, World Statistics, https://www.fao.org/faostat/en/#data/GT
[^22]: United Nations, https://www.un.org/en/observances/end-food-waste-day
[^23]: Le monde, https://www.lemonde.fr/planete/article/2013/02/28/la-viande-de-boeuf-dans-votre-assiette-de-la-vieille-vache_1839589_3244.html
[^24]: OurWorldInData: Rise of Aquaculture, https://ourworldindata.org/rise-of-aquaculture
[^25]: Food and Agriculture Organisation, https://www.fao.org/3/cc8543en/cc8543en.pdf
[^26]: Food and Agriculture Organisation, https://www.fao.org/platform-food-loss-waste/flw-data/en