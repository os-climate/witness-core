The Earth offers a finite amount of different type of lands which can be used to build city, to grow forest... or can not be used at all (ocean or glacier for example).
The land use model focuses on agricultural and forest land, which is shared between different activities: 
* Land usage for food from crop
* Land usage for food from livestock
* Energy linked technologies land for crop
* Energy linked technologies forest land


### Model data

The following data (taken from 'Our World in Data'[^1]) are integrated into the model:

|Category|Name|Surface|Magnitude|Unit|
| ------ | -- |:-----:|:-------:|:--:|
|Earth|Land|149|M|km2|
|Earth|Ocean|361|M|km2|
|Land|Habitable|105.79|M|km2|
|Land|Glacier| 4.9|M|km2|
|Land|Barren|28.31|M|km2|
|Habitable|Agriculture|52.89|M|km2|
|Habitable|Forest|39.14|M|km2|
|Habitable|Shrub|11.64|M|km2|
|Habitable|Urban|1.06|M|km2|
|Habitable|Water|1.06|M|km2|
|Agriculture|Livestock|40.73|M|km2|
|Agriculture|Crops|12.17|M|km2|

These data give the share of Earth land. The focus will be done on agriculture land.
First of all, as priority is given to food, land will be used to feed humanity.
Food and Agriculture Organisation[^2] gives the average land use per person, per year.
Quote: "About one-third of this is used as cropland, while the remaining two-thirds consist of meadows and pastures for grazing livestock."
In 2016, **0.21 hectare per person** were used to feed population with crop.
As a result, **0.42 hectare per person** were used to feed population with livestock.
Multiplying these values by the population (number of people on planet) gives the surface needed to feed humanity.

Then, the formula is:
$$Land\_Use\_Crop\_Food = Average\_Land\_Use\_Crop\_Per\_Person * Population$$
$$Land\_Use\_Livestock\_Food = Average\_Land\_Use\_Livestock\_Per\_Person * Population$$
$$Land\_Use\_Food = Land\_Use\_Crop\_Food + Land\_Use\_Livestock\_Food$$
$$Available\_Land\_Agriculture= Total\_Agriculture\_Land - Land\_Use\_Food$$

This Available\_Land\_Agriculture can now be used by energy linked technologies.

Technologies taking surface of agriculture land are:
* Crop energy
* Solar photovoltaic

Technologies taking surface of forest land are:
* Managed wood
* Unmanaged wood
* Reforestation


A usage factor is added to control the usage of land for food from livestock. This factor permit in a very simple way to introduce a diet variation, and to allow the model to give more place for energy technologies.

### References 

[^1]: Hannah Ritchie and Max Roser (2013) - "Land Use". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/land-use'
[^2]: Food and Agriculture Organization of the United Nations (2020) - "Land use in agriculture by the numbers". Published online at fao.org. Retrieved from ':http://www.fao.org/sustainability/news/detail/en/c/1274219/'