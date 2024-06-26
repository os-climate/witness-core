# IEA Data preparation Model

The model is designed to handle the interpolation and extrapolation of various energy-related datasets. The model processes multiple DataFrames representing different energy production, energy prices, CO2 taxes, population, and temperature, as provided by the International Energy Agency (IEA).
The model computes the "missing" data to get a yearly value

## Processing IEA input data

The ultimate goal consists in comparing IEA data with witness data. However, IEA an witness do not always provide the same quantities
and some data processing is required. This section describes how the IEA data have been processed.

- **population**: data are directly taken from [^1]
- **GDP**: data are directly taken from [^2]
- **CO2 price**: IEA report provides the average CO2 price for categories of countries [^1]: 1) Advanced economies with
  net zero emissions pledges, 2) Emerging market and developing economies with net zero emissions pledges, 3) Selected
  emerging market and developing economies (without net zero emissions pledges) and 4) Other emerging market and developing
  economies. Then, in order to compute an average worldwide CO2 price, the CO2 price of each category is weighted by the share of
  world GDP of this category. Indeed, more than 140 countries have pledged to be net zero emissions which covers 88% of
  the global CO2 emissions [^3]. Therefore, as a first order of magnitude, assuming that global contribution (in %)
  to CO2 emissions and GDP are equal for a given country, then it is assumed that 12% of the GDP of emerging market and
  developing economies account for countries that have not pledged for net zero emissions. Arbitrarily (no data available),
  it is assumed that half of that GDP is equally split between categories 3) and 4). Furthermore, this ratio of GDP
  between countries is assumed to be constant until 2050, which is a highly debatable hypothesis. Since IEA data start in 2030
  for the CO2 price, then value for 2020 is taken from witness directly
- **CO2 emissions**: data from [^1] were extracted using [^4] from 2022 to 2050 with a step of 5 years
- **Atmospheric Temperature**: data from [^5] were extracted using [^4] from 2022 to 2100 with a step of 5 years
- **Energy production**: Data are taken from [^2], fig 02_05. The values correspond to the raw energy of a given techno.
  For nuclear for instance, this corresponds to the TWh of electricity and heat produced. Therefore, the witness data of
  heat and electricity must be added for nuclear to be comparable with the IEA data. Same goes with wind energy where
  witness offshore and onshore wind energy data must be added to be compared to the global IEA windenergy data.
  Similarly, the witness solar pv and solar thermal must be added to be compared to the IEA data. Evenutally, for coal, fuel,
  IEA data correspond to the energy associated to the Mt of coal and fuel extracted. For the bioenergy, IEA splits into
  liquid, solid and traditional use of biomass. In order to compare witness and IEA, the witness crop model energy production
  is compared with the sum of the IEA energy production in "Conventional bioenergy crops" and "Short-rotation woody crops".
  Similarly, the witness forest model energy production is compared with the sum of the IEA energy production in
  "Traditional use of biomass", "Forestry plantings" and "Forest and wood residues".
- **Energy prices**: only the electricity prices are compared between witness and IEA. IEA provides data for 4 regions
  (USA, China, EU, India) [^2]. Then the average electricity price per techno is obtained by weighting the price of each region
  by the GDP share of that region. This is an approximation since the sum of the GDP shares of those regions amounts for
  64% of the world GDP in 2022. It is therefore debatable to consider that this electricity price is representative of the world's
  average electricity price. This however allows to make a first comparison.

Eventually, the data processed as above are saved under witness-core\climateeconomics\data\IEA_NZE\*.csv. In the csv,
the column name is identical to the corresponding witness variable name. Furthermore, the name of the corresponding witness dataframe
that the variable is taken from is described in the csv file name. This is however not applied to quantities that
require for instance summing of several witness variables. In this case, the variable name is just "Nuclear" for instance.

## Input Data

The input to the model consists of a dictionary of DataFrames. Each DataFrame represents a specific dataset provided by the IEA and includes:

- **Years**: A column representing the year for each data point.
- **Values**: Columns representing various metrics such as production quantities, prices, taxes, population, and temperature.

Example DataFrames:

- DataFrame for energy production of various technologies.
- DataFrame for energy prices.
- DataFrame for CO2 tax rates.
- DataFrame for population data.
- DataFrame for temperature data.

## Output Data

The output of the model is a dictionary of DataFrames similar to the input, but with the following modifications:

- Output variable have the same format as input, the name of each output is the name of the corresponding input with
  a "\_interpolated" suffix.
- The 'years' column spans from `year_start` to `year_end`.
- The values in the other columns are interpolated linearly for the years within the provided data range.
- If the data in any DataFrame starts after `year_start`, the model performs backward extrapolation to estimate values for the missing initial years.

## The Model

The model performs the following steps for each DataFrame in the input dictionary:

1. **Linear Interpolation**:

   - Apply linear interpolation to fill in the missing values within the range of available data points.

2. **Backward Extrapolation**:
   - Identify the first valid data point in each column.
   - Calculate the slope based on the first two valid data points.
   - Use this slope to perform linear backward extrapolation, estimating values for the years before the first valid data point.

## References

[^1]: IEA Global Energy and Climate model documentation 2023, <https://iea.blob.core.windows.net/assets/ff3a195d-762d-4284-8bb5-bd062d260cc5/GlobalEnergyandClimateModelDocumentation2023.pdf>

[^2]: IEA Net Zero by 2050, A roadmap for the Energy sector, <https://iea.blob.core.windows.net/assets/deebef5d-0c34-4539-9d0c-10b13d840027/NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf>

[^3]: UN climate change coalition, <https://www.un.org/en/climatechange/net-zero-coalition#:~:text=Yes%2C%20a%20growing%20coalition%20of,about%2088%25%20of%20global%20emissions>.

[^4]: Plot digitizer, <https://plotdigitizer.com/app>

[^5]: IEA data and statistics, oct 2021, <https://www.iea.org/data-and-statistics/charts/global-median-surface-temperature-rise-in-the-weo-2021-scenarios-2000-2010>
