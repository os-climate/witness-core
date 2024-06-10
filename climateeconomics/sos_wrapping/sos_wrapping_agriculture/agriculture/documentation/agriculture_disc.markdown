# Agriculture Model

Food is one of the primary needs of human being. However, it is also a real challenge to produce food with the limited space of the land on our planet, and the important population.
The agriculture model aims to answer to the following question:
Given a food diet for humanity, how much space do we need to to feed everyone ?
This model allows to test different food diets, and computes the agriculture surface needed to produce enough food to feed all human beings.

## Model data

This section lists the inputs and outputs of the model.

The model inputs are:

- **year start**, this defines the starting year of the study.
- **year end**, this defines the last year of the study.
- **population_df**, this gives the population number, for each year of the study. The unit is the million of people.
- **temperature_df**, gives the temperature in degree Celsius above pre-industrial level for each year of the study.
- **starting diet**, this gives the food diet for the first year of the study. For each food, the quantity in kg/person/year need to be filled. The different food taken into account are:
  red meat, white meat, milk, eggs, rice and maize, potatoes, fruits and vegetables. By default, data are:

| Red meat | White meat | Milk  | Eggs | Rice and Maize | Potatoes | Fruit and vegetables | Unit           |
| -------- | :--------: | ----- | ---- | :------------: | :------: | :------------------: | -------------- |
| 11.02    |   31.11    | 79.27 | 9.68 |     97.76      |  32.93   |        217.62        | kg/person/year |

These data are the average world diet [^1].

- **m2 per kg**, gives the average area needed to produce 1kg of the considered food. The unit is m^2/kg. By default, data comes from [^2].
- **kcal per kg**, gives the number of kilocalories for 1kg of the considered food. The unit is kcal/kg.
  By default, the values are:

| Red meat | White meat | Milk | Eggs | Rice and Maize | Potatoes | Fruit and vegetables | Unit    |
| -------- | :--------: | ---- | ---- | :------------: | :------: | :------------------: | ------- |
| 2566     |    1860    | 550  | 1500 |      1150      |   670    |         624          | kcal/kg |

These data are extracted from [^3].

- **red meat calory percentage**, gives the percentage of red meat kcal in a person diet.
- **white meat calory percentage**, gives the percentage of white meat kcal in a person diet.
- **other_use_agriculture**, gives the average ha per person for the use of agriculture in other way that the 7 food types. It mainly takes into account :
  cocoa - coffee - olive - sugar - oil(palm, sunflower,...) - tea - grapes(wine) - tobacco - yams - natural rubber - millet - textile fiber (cotton and other)...
  By default, it is set to 0.102ha/person to represent around of 800 millions of ha for a population of 7.8billions of people. [^4]

The model outputs are:

- **agriculture productivity evolution**, gives the evolution of agriculture productivity overtime, due to climat change. Unit is %.
- **food land surface**, gives the surface needed to produce the amount of food of the diet. The unit is Gha.
- **food land surface percentage**, gives the share of the surface of each type of food, in %.
- **final diet**, gives the diet evolution over the years, in kg/person/year, for each type of food.
- **Total food land surface**, gives the total amount of land needed for food, in Gha.

## From a food diet to a surface

This section aims to explain how the model goes from a food diet to a surface to produce the required food.
The following picture shows the "how to" process.
![](diet_to_surface.PNG)

First, given the population and the diet per person, the quantity of food is deduced:
$$quantity\_of\_food = population * diet\_per\_person$$

Then, knowing the conversion factor between mass of food and surface needed to produce it, the required surface is given by:
$$required\_surface = quantity\_of\_food * square\_meter\_per\_food$$

To all of this, an other contribution will be added to the agriculture surface. This other contribution represents the use of agriculture for other purpose than producing the 7 food types already taken into account. It mainly takes into account :
cocoa - coffee - olive - sugar - oil(palm, sunflower,...) - tea - grapes(wine) - tobacco - yams - natural rubber - millet - textile fiber (cotton and other) - wheat...

## Changing the diet

This section aims to explain how the diet change works in the agriculture model.

In this model, the user is able to change diet over time, then, to observe the effect of the changes.
First of all, when a diet change, one of the risk is to be undernourished. To prevent that, the diet will remain at the same amount of kcal all the time.
When the term "convert" a food X to an other Y is used, it means suppress a quantity of food X, which corresponds to an amount of E kcal, and then add a amount of food Y that fill the E kcal removed.

The model starts from the base-diet given in input, **diet_df**.
The first change is to convert red meat and white meat, using **red_meat_percentage** and **white_meat_percentage** to vegetables (fruit and vegetables, potatoes and rice and maize).
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

This red_meat_removed corresponds to a quantity of kcal determined by:
$$red\_energy\_removed = base\_diet\_red\_meat - total\_kcal * red\_meat\_percentage / 100$$

Then, the new quantity of white meat is calculated:
$$base\_diet\_white\_meat = total\_kcal * white\_meat\_percentage / 100 / kg\_to\_kcal\_white\_meat$$

This white_meat_removed corresponds to a quantity of kcal determined by:
$$white\_energy\_removed = base\_diet\_white\_meat - total\_kcal * white\_meat\_percentage / 100$$

Finally, the diet has to be updated with the conversion into vegetables:
the amount of calory is proportionaly reported to the vegetables.
for the 'fruit and vegetables' diet:
$$proportion = base\_diet\_kcal\_fruit\_and\_vegetables \\/ (base\_diet\_kcal\_fruit\_and\_vegetables + base\_diet\_kcal\_potatoes + base\_diet\_kcal\_rice\_and\_maize)$$
$$base\_diet\_fruit\_and\_vegetables += (red\_energy\_removed + white\_energy\_removed) \\* proportion / kg\_to\_kcal\_fruit\_and\_vegetables$$

The same formula is applicated to the 'potatoes' and 'rice and maize' food categories.

## Climate change impact

The increase in temperature due to global warming has consequences on crop productivity. Effects of global warming on agriculture are for example drought, flooding and increased crops water needs.
Some models estimate the impact of increased temperature on crop productivity but results are quite disparate. We therefore chose to implement one impact for all agriculture productivity and not one per type of crop (maize, rice, wheat...). Our bilbiography to define the effect includes IPCC[^5] (Intergovernmental Panel on Climate Change), the FAO[^6] (Food and Agriculture Organization), Carl-Friedrich Schleussner et al (2018)[^8] and Rosenzweig et al. (2014)[^7].
The surface required taking into account climate change is:
$$Required\_surface\_with\_global\_warming = required\_surface * (1- productivity\_change)$$
$$Productivity\_change = \alpha T^2+bT$$
$$with~~T_t = temperature_t -temperature_{t=0}$$
By default $\alpha$ and $b$ are respectively set at -0.00833 and - 0.04167. So that at +2 degrees (above pre industrial level) the decrease in productivity is 5% and 30% at +5 degrees celsius.
Then, when temperature rises, the surface required to produce the same amount of food increases by the share of production reduction.

## Results

This sections aims to describes the results charts.

- Surface taken to produce food over time:
  This is a bar graph, displaying the surface needed (Gha) over the time (years of the study).
  The detail is given for each food considered. Plus, a line plots the total surface needed to feed humanity, with the considered diet.
- Share of the surface used to produce food over time:
  This graph display the same information than the last one, but values are normalized in percentage.
- evolution of the diet overtime:
  This graphs gives the diet evolution for each year of the study, in kg / person / year, for each food.

## Model limitation

This section aims to gives the limitations of the model.

The agriculture model gives output data as the result of different computations described previously.
However, the different hypothesis lead to limitations that should be kept in mind in order to have a critical point of view regarding results.
The following points list the limitations of the agriculture model:

- The model does not allow a modification of the global kcal amount of the diet. This kcal amount is set by the initial diet, and will remain the same.
- The model does not test the feasibility of the diet change. For example, convert 100% of red meat to white meat in one year is allowed in the model, even if it is probably not realisable.
- The model considers a global average diet, and does not include change depending on country or habits.
- The land use to produce food are assumed to be able to produce any type of food, without regarding average weather or climate conditions. For example equatorial or temperate climate may have different affinity with food production.

## Example of diet

This section gives example of diet data that can be used as diet_df input.
Data are extracted from [^1], and are given in kg/person for the year 2018

| Region | Red meat | White meat | Milk  | Eggs   | Rice and Maize | Potatoes | Fruit and vegetables | Unit      | Year |
| ------ | -------- | :--------: | ----- | ------ | :------------: | :------: | :------------------: | --------- | ---- |
| World  | 11.02    |   31.11    | 79.27 | 9.68   |     97.76      |  32.93   |        217.62        | kg/person | 2018 |
| USA    | 37.7     |   84.63    | 223.7 | 16.21  |     22.99      |  52.26   |        203.07        | kg/person | 2018 |
| China  | 8.97     |   52.51    | 23.13 | 19.74  |     125.47     |   41.9   |        463.29        | kg/person | 2018 |
| France | 23.43    |   54.44    | 185   | 211.72 |     20.06      |  49.18   |        181.92        | kg/person | 2018 |

## References

[^1]: FAO, FAOSTAT, Food Balances (2014-), Latest update: April 14 2021, Accessed August 2021, <http://www.fao.org/faostat/en/#data/FBS>

[^2]: Poore, J., & Nemecek, T. (2018). Reducing food's environmental impacts through producers and consumers. Science, 360(6392), 987-992. Published online at OurWorldInData.org. Retrieved from: <https://ourworldindata.org/grapher/land-use-per-kg-poore>

[^3]: FAO, Food Balance Sheets A handbook, Annex 1 : Food Composition Tables, <http://www.fao.org/3/X9892E/X9892e05.htm#P8217_125315>

[^4]: Knoema, Production Statistics - Crops, Crops Processed, <https://knoema.com/FAOPRDSC2020/production-statistics-crops-crops-processed?country=1002250-world>

[^5]: IPCC, 2007. Climate Change 2007: Impacts, Adaptation and Vulnerability Contribution of Working Group II to the Fourth Assessment. Report of the Intergovernmental Panel on Climate Change, M.L. Parry, O.F. Canziani, J.P. Palutikof, P.J. van der Linden and C.E. Hanson, Eds., Cambridge University Press, Cambridge, UK, 976pp.

[^7]: Rosenzweig, C., Elliott, J., Deryng, D., Ruane, A.C., Müller, C., Arneth, A., Boote, K.J., Folberth, C., Glotter, M., Khabarov, N. and Neumann, K., 2014. Assessing agricultural risks of climate change in the 21st century in a global gridded crop model intercomparison. Proceedings of the national academy of sciences, 111(9), pp.3268-3273.

[^6]: FAO, 2019. The State of Food and Agriculture 2016 (SOFA): Climate Change, Agriculture and Food Security.

[^8]: Schleussner, C.F., Deryng, D., Muller, C., Elliott, J., Saeed, F., Folberth, C., Liu, W., Wang, X., Pugh, T.A., Thiery, W. and Seneviratne, S.I., 2018. Crop productivity changes in 1.5 C and 2 C worlds under climate sensitivity uncertainty. Environmental Research Letters, 13(6), p.064007.
