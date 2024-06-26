About 71 % of the Earth's surface is covered with water, and the remaining lands are composed of 71 % of what we call habitable lands (i.e. usable for human activities), 10 % glacier and 19 % barren land.

The land use model focuses on habitable lands, which are shared between different activities:

- Land usage for food from crop and livestock
- Energy linked technologies land

The initial land distribution for habitable lands (50% for agriculture and 37% for forest in 2019), may vary with deforestation and reforestation.

The model's inputs are therefore:

- Food land surface coming from agriculture discipline
- Energy linked technologies and reforestation surface (for CCS purpose) coming from the energy mix discipline
- Deforestation surface coming from forest discipline

For each year, the available forest and agricultural surface is computed with deforestation and reforestation inputs, and a constraint is calculated for each such as:

$$Agriculture\_demand\_constraint = available\_agriculture\_surface + deforested\_surface \\ - reforested\_surface\_for\_CCUS - surface\_used\_for\_food - surface\_for\_energy$$
$$Forest\_demand\_constraint = available\_forest\_surface - deforested\_surface \\ + reforested\_surface\_for\_CCUS - surface\_for\_energy$$

So that the optimizer will try to minimize the absolute value of the constraint to solve the optimization problem.

### Model's data

The following data (taken from 'Our World in Data'[^1]) are integrated into the model (for the 2019 year):

| Category    | Name        | Surface | Magnitude | Unit |
| ----------- | ----------- | :-----: | :-------: | :--: |
| Earth       | Land        |   149   |     M     | km2  |
| Earth       | Ocean       |   361   |     M     | km2  |
| Land        | Habitable   |   104   |     M     | km2  |
| Land        | Glacier     |   15    |     M     | km2  |
| Land        | Barren      |   28    |     M     | km2  |
| Habitable   | Agriculture |   51    |     M     | km2  |
| Habitable   | Forest      |   39    |     M     | km2  |
| Habitable   | Shrub       |   12    |     M     | km2  |
| Habitable   | Urban       |   1.5   |     M     | km2  |
| Habitable   | Water       |   1.5   |     M     | km2  |
| Agriculture | Livestock   |   40    |     M     | km2  |
| Agriculture | Crops       |   11    |     M     | km2  |

### References

[^1]: Hannah Ritchie and Max Roser (2013) - "Land Use". Published online at OurWorldInData.org. Retrieved from: '<https://ourworldindata.org/land-use>'
