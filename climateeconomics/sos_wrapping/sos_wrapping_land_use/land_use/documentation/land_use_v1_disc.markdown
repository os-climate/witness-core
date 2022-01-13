The Earth offers a finite amount of different type of lands which can be used to build city, to grow forest... or can not be used at all (ocean or glacier for example).
The land use model focuses on agricultural and forest land, which is shared between different activities: 
* Land usage for food from crop
* Land usage for food from livestock
* Energy linked technologies land for crop
* Energy linked technologies forest land


### Model's data

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

These data give the share of Earth land.

This Available\_Land\_Agriculture can be used by energy linked technologies.

Technologies taking surface of agriculture land are:
* Crop energy
* Solar photovoltaic

Technologies taking surface of forest land are:
* Managed wood
* Unmanaged wood

A particular case for the Reforestation Technology: this technology changes constraints on available lands surface.
It adds surface on forest available land and removes surface from agriculture available land.

### References 

[^1]: Hannah Ritchie and Max Roser (2013) - "Land Use". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/land-use'