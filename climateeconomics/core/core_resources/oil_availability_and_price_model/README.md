# WITNESS - Oil resource model - Data Analyst Nanodegree final project

## Purpose

Build a model that predict trends for availability and price of oil until 2100. This model shall be integratedto the energy and economics model for example to impact the choice on energy mix considering the limited resources on earth.

## Approach

This model is designed in several stages that answers the following questions

1. Will we reach a limit to the oil production vs the demand? When will be this mawimum production? When could shortage in oil supply vs demand occur?
   1. What are the reserves of oil?
   2. How much is extracted and predicted to be extracted?
   3. How is the demand?
2. How will rise the oil price around shortage?
   1. What will be the volume extracted by type of exration?
   2. What is the cost of extraction?
   3. What is the price of transportation?
   4. What is the part in the prediction related to offer/demand inflation?

## Inputs

- date where to make the prediction
- optional - volume demand in oil - default shall be the pridiction of oil demand
- region or global on which to provide the result
- optional - type of extraction on which to focus - list

## Ouputs

- oil price for a given year or other time period to adjust
- oil volume available in time
- breakdown of oil price (extraction, transport, offer/demand)

## Data for extrapolation and correlation

### Summary

This table sums up the found sources vs the model needs.

| Required data | Used for - model | Dataset ID/name list | Characteristics required | Prefered dataset |
|:---------------|:----------------:|:--------------------:|:------------------------:|-----------------:|
|Crude oil sell price| oil demand prediction, oil price prediction|2 - Oil price||
|Crude oil extraction price| oil price prediction|
|Crude oil transport price| oil price prediction|
| Crude oil volume extraction | 
| Crude oil reserves |||by type of extraction / by region||
| Crude oil demand||11 - Crude Oil Dataset||11 - Crude Oil Dataset

### 0 - BP sat reviews 2020

_Data for crude oil and crude oil price_

> see definition in excel file: ```data/bp-stats-review-2020-all-data.xlsx```

```
data/bp-stats-review-2020-consolidated-dataset-panel-format.csv
data/bp-stats-review-2020-all - Oil - Crude prices since 1861.csv
```

https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy/oil.html#oil-reserves


**most relevant daatsource for all informations**

### 1 - Oil extraction:

>The industrial production (IP) index measures the real output of all relevant establishments located in the United States, regardless of their ownership, but not those located in U.S. territories.

https://fred.stlouisfed.org/series/IPG211111CS

>Board of Governors of the Federal Reserve System (US), Industrial Production: Mining, Quarrying, and Oil and Gas Extraction: Crude Oil (NAICS = 211111pt.) [IPG211111CS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/IPG211111CS, July 16, 2021.

*__not useful for current study - the index reflect not ony the production of crude oil and is not comparable ith other data because the volume is not expressed__*

### 2 - Oil price:

>In the context of The Global dataset of oil and natural gas production, prices, exports, and net exports with Oil production and prices data are for 1932-2014 (2014 data are incomplete); Gas production and prices are for 1955-2014; Export and net export data are for 1986-2013 data are given to us from here we need to analyse the historical prices and to predict the future price.

https://www.kaggle.com/c/predicting-oil-price/

*__Interesting for: oil production, oil sell price and oil export by country__*

### 3 - Crude oil import prices

> Crude oil import prices come from the IEA's Crude Oil Import Register and are influenced not > only by traditional movements of supply and demand, but also by other factors such as geopolitics. Information is collected from national agencies according to the type of crude oil, by geographic origin and by quality of crude. Average prices are obtained by dividing value by volume as recorded by customs administrations for each tariff position. Values are recorded at the time of import and include cost, insurance and freight, but exclude import duties. The nominal crude oil spot price from 2003 to 2011 is for Dubai and from 1970 to 2002 for Arabian Light. This indicator is measured in USD per barrel of oil. The real price was calculated using the deflator for GDP at market prices and rebased with reference year 1970 = 100.
> 
https://data.oecd.org/energy/crude-oil-import-prices.htm#indicator-chart

```
DP_LIVE_16072021143520793.csv
```

> OECD (2021), Crude oil import prices (indicator). doi: 10.1787/9ee0e3ab-en (Accessed on 16 July 2021)

### 4 - Crude oil production

>Crude oil production is defined as the quantities of oil extracted from the ground after the removal of inert matter or impurities. It includes crude oil, natural gas liquids (NGLs) and additives. This indicator is measured in thousand tonne of oil equivalent (toe).Crude oil is a mineral oil consisting of a mixture of hydrocarbons of natural origin, yellow to black in colour, and of variable density and viscosity. NGLs are the liquid or liquefied hydrocarbons produced in the manufacture, purification and stabilisation of natural gas. Additives are non-hydrocarbon substances added to or blended with a product to modify its properties, for example, to improve its combustion characteristics (e.g. MTBE and tetraethyl lead).Refinery production refers to the output of secondary oil products from an oil refinery.

https://data.oecd.org/energy/crude-oil-production.htm#indicator-chart

```
DP_LIVE_16072021143543731.csv
```

>OECD (2021), Crude oil production (indicator). doi: 10.1787/4747b431-en (Accessed on 16 July 2021)

### 5 - Oil and Gas Summary Production Data: 1967-1999

>This dataset contains production information from oil and gas wells in New York State from 1967 to 1999. Each record represents a sum by operator for each county, town, field, and formation grouping.

https://data.ny.gov/d/8y5c-ebxg

https://data.world/data-ny-gov/8y5c-ebxg

```
oil-and-gas-summary-production-data-1967-1999-1.csv
```

>Last updated at https://data.ny.gov/data.json : 2019-06-10

#### 6 - Oil, Gas, & Other Regulated Wells: Beginning 1860

> Information on oil, gas, storage, solution salt, stratigraphic, and geothermal wells in New York State

https://data.ny.gov/d/szye-wmt3
Last updated at https://data.ny.gov/data.json : 2020-09-14
https://data.world/data-ny-gov/szye-wmt3

```
oil-gas-other-regulated-wells-beginning-1860-1.csv
```

### 7 - US Distribution Production Oil

>Distribution tables of oil and gas wells by production rate for all wells, including marginal wells, are available from the EIA for most states for the years 1919 to 2009. Graphs displaying historical behavior of well production rate are also available. The quality and completeness of data is dependent on update lag times and the quality of individual state and commercial source databases. Undercounting of the number of wells occurs in states where data is sometimes not available at the well level but only at the lease level. States not listed below will be added later as data becomes available.

https://catalog.data.gov/dataset/distribution-and-production-of-oil-and-gas-wells-by-state
https://data.world/doe/us-distribution-production-oil

### 8 - Investment, production and operational costs - not accessible

![https://www.rystadenergy.com/contentassets/51bfbb0ef64b417cb516a29d1e5b9ef7/upstream_ucube_sb1-prop.jpg](https://www.rystadenergy.com/contentassets/51bfbb0ef64b417cb516a29d1e5b9ef7/upstream_ucube_sb1-prop.jpg)

https://www.rystadenergy.com/energy-themes/oil--gas/upstream/u-cube/

### 9 - Brent oil price

>The aim of this dataset and work is to predict future Crude Oil Prices based on the historical data available in the dataset.
The data contains daily Brent oil prices from 17th of May 1987 until the 25th of February 2020.

https://www.kaggle.com/mabusalah/brent-oil-prices

Europe Brent Spot Price FOB (Dollars per Barrel)
https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls

```
BrentOilPrices.csv
```

*__it collect price data per each day of the stock exchange pricing but relates to brent oil instead of crude oil__*

### 10 - Timeseries Crude Oil Production by Country

>This dataset about Crude Oil Production by Country is extracted from Flourish visualisation. If you want to know more about Flourish click here.

https://www.kaggle.com/mathurinache/crudeoilproductionbycountry

```
Crude Oil Production by Country
```


### 11 - Crude Oil Dataset

>Context
Crude Oil is one of the most important commodities in the world. Hence in order to preserve it for the future, proper analysis is essential. This dataset would help in the analysis of it's price variation.

>Content
What's inside is more than just rows and columns. Make it easy for others to get started by describing how you acquired the data and what time period it represents, too.

https://www.kaggle.com/dharinir242001/crude-oil-dataset?select=CRUDE_OIL.csv

```
CRUDE_OIL.csv
```

*__Interesting for the volume/ price  per day to assess the demand vs price: use the price as input and predict the associated volume of demand and adjust with the volume demand trend as basis__*

### 12 - Costs of Crude Oil and Natural Gas Wells Drilled

>Real costs are in chained (2000) dollars, calculated by using gross domestic product price deflators.
The information reported for 1965 and prior year is not strictly comparable to that in more recent surveys.
Average cost is the arithmetic mean and includes all costs for drilling and equipping wells and for surface-producing facilities.
Wells drilled include exploratory and development wells; excludes service wells, stratigraphic tests, and core tests.

|Key Terms|	Definition|
|----|-----|
|Crude Oil	|A mixture of hydrocarbons that exists in liquid phase in natural underground reservoirs and remains liquid at atmospheric pressure after passing through surface separating facilities. Depending upon the characteristics of the crude stream, it may also include: - Small amounts of hydrocarbons that exist in gaseous phase in natural underground reservoirs but are liquid at atmospheric pressure after being recovered from oil well (casinghead) gas in lease separators and are subsequently commingled with the crude stream without being separately measured. Lease condensate recovered as a liquid from natural gas wells in lease or field separation facilities and later mixed into the crude stream is also included -Small amounts of nonhydrocarbons produced with the oil, such as sulfur and various metals; - Drip gases, and liquid hydrocarbons produced from tar sands, oil sands, gilsonite, and oil shale. Liquids produced at natural gas processing plants are excluded. Crude oil is refined to produce a wide array of petroleum products, including heating oils; gasoline, diesel and jet fuels; lubricants; asphalt; ethane, propane, and butane; and many other products used for their energy or chemical content.|
|Development Well	|A well drilled within the proved area of an oil or gas reservoir to the depth of a stratigraphic horizon known to be productive.
|Dry Hole|	An exploratory or development well found to be incapable of producing either oil or gas in sufficient quantities to justify completion as an oil or gas well.
|Exploratory Well|	A hole drilled: a) to find and produce oil or gas in an area previously considered unproductive area; b) to find a new reservoir in a known field, i.e., one previously producing oil and gas from another reservoir, or c) to extend the limit of a known oil or gas reservoir.
|Natural Gas|	A gaseous mixture of hydrocarbon compounds, primarily methane, used as a fuel for electricity generation and in a variety of ways in buildings, and as raw material input and fuel for industrial processes.
|Well|	A hole drilled in the earth for the purpose of (1) finding or producing crude oil or natural gas; or (2) producing services related to the production of crude or natural gas.

https://www.eia.gov/dnav/pet/pet_crd_wellcost_s1_a.htm

```
PET_CRD_WELLCOST_S1_A.xls
```

*__interesting for investment and extraction price__*

### 13 OECD stats Material Resources >

>ey statistical concept
Contact: ENV.Stat@oecd.org 

Last update: March 11, 2021

The data presented come from the UNEP "Environment Live" database (http://uneplive.unep.org/material ; for non_EU countries and for material footprint data) and from Eurostat's "Material Flow and Productivity" database (https://ec.europa.eu/eurostat/web/environment/material-flows-and-resource-productivity/database, for EU countries+ Norway, Switzerland, North Macedonia, Albania, Serbia, Turkey and Bosnia Herzegovina). Data for EU countries exclude the material group "other" (as presented in Eurostat's database). Country aggregates include intra-trade.

It should be borne in mind that the data should be interpreted with caution and that the time series presented here may change in future as work on methodologies for MF accounting progresses. Furthermore, data contain rough estimates for OECD and BRIICS aggregates.

These data refer to material resources, i.e. materials originating from natural resources that form the material basis of the economy: metals (ferrous, non-ferrous) non-metallic minerals (construction minerals, industrial minerals), biomass (wood, food) and fossil energy carriers.

The use of materials in production and consumption processes has many economic, social and environmental consequences. These consequences often extend beyond the borders of countries or regions, notably when materials are traded internationally, either in the form of raw materials or as products embodying them. They differ among the various materials and among the various stages of the resource life cycle (extraction, processing, use, transport, end-of-life management). From an environmental point of view these consequences depend on:

the rate of extraction and depletion of renewable and non-renewable resource stocks
the extent of harvest and the reproductive capacity and natural productivity of renewable resources
the associated environmental burden (e.g. pollution, waste, habitat disruption), and its effects on environmental quality (e.g. air, water, soil, biodiversity, landscape) and on related environmental services
These data inform about physical flows of material resources at various levels of detail and at various stages of the flow chain. The information shows:

a) the material basis of economies and its composition by major material groups, considering:

the extraction of raw materials;
the trade balance in physical terms;
the consumption of materials;
the material inputs
b) the consumption of selected materials that are of environmental and economic significance.

c) in-use stocks of selected products that are of environmental and economic significance.

Domestic extraction used (DEU) refers to the flows of raw materials extracted or harvested from the environment and that physically enter the economic system for further processing or direct consumption (they are used by the economy as material factor inputs).

Imports (IMP) and exports (EXP) are major components of the direct material flow indicators DMI (domestic material input) and DMC (domestic material consumption). They cannot be taken as indication of domestic resource requirements.

Domestic material consumption (DMC) refers to the amount of materials directly used in an economy, which refers to the apparent consumption of materials. DMC is computed as DEU minus exports plus imports.

Domestic material input (DMI) is computed as DEU plus imports.

Material Footprint (MF) refers to the global allocation of used raw material extracted to meet the final demand of an economy.

The material groups are:

Food: food crops (e.g. cereals, roots, sugar and oil bearing crops, fruits, vegetables), fodder crops (including grazing), wild animals (essentially marine catches), small amounts of non-edible biomass (e.g. fibres, rubber), and related products including livestock.

Wood: harvested wood and traded products essentially made of wood (paper, furniture, etc.).

Construction minerals: non-metallic construction minerals whether primary or processed. They comprise marble, granite, sandstone, porphyry, basalt, other ornamental or building stone (excluding slate); chalk and dolomite; sand and gravel; clays and kaolin; limestone and gypsum.

Industrial minerals: non-metallic industrial minerals whether primary or processed (e.g. salts, arsenic, potash, phosphate rocks, sulphates, asbestos).

Metals: metal ores, metals and products mainly made of metals.

Fossil fuel: coal, crude oil, natural gas and peat, as well as manufactured products predominantly made of fossil fuels (e.g. plastics, synthetic rubber).

https://stats.oecd.org/Index.aspx?DataSetCode=MATERIAL_RESOURCES

```
data/MATERIAL_RESOURCES_27072021094049150.csv
```

### 14 OECD stats Mineral and Energy Resources

>
Click to collapse Contact person/organisation
sdd.seea@oecd.org
Click to collapse Date last input received
26 May 2020

Click to collapse Population & Scope
Click to collapse Population coverage
14 commodities were selected based on their economic and environmental significance. They are listed below:
Crude oil (billion barrels)
Natural gas (billion cubic metres)
Hard coal (billion tonnes)
Brown coal (billion tonnes)
Coal (billions tonnes)
Iron-Ore (billion tonnes)
Bauxite (billion tonnes)
Copper (million tonnes)
Tin (thousand tonnes)
Zinc (million tonnes)
Lead (million tonnes)
Nickel (million tonnes)
Gold (tonnes)
Silver (thousand tonnes)
Phosphate (million tonnes)

9 OECD countries possessing a significant share of these key commodities and for which data and metadata were available have been included in the database so far: Australia, Canada, Colombia, Denmark,  Mexico, the Netherlands, Norway, the United Kingdom and the United States.

Click to collapse Concepts & Classifications
Click to collapse Key statistical concept
More detailed information available in the working paper "Compiling mineral and energy resource accounts according to the System of Environmental-Economic Accounting (SEEA) 2012"

 

Click to collapse Key statistical concept
 

Mineral and energy resources are one of the seven environmental assets considered in the System of Environmental Economic Accounting (SEEA, 2012). They are non-renewable resources which cannot be regenerated over a human timescale in spite of their prominent role in sustaining economic activities. From an economic, environmental and supply security perspective, it is therefore important to gather harmonised data on their rate of extraction and current availability.

Stocks and flows for each commodity are compiled in physical units and three classes of resources are distinguished, as advocated by the SEEA (2012): commercially recoverable resources (Class A), potentially commercially recoverable resources (Class B) and non-commercial and other known deposits (Class C). The definition of these classes is based on the United Nations Framework Classification for Fossil Energy and Mineral Reserves and Resources (UNFC 2009) which, in turn, can be related to two other major classification systems: CRIRSCO for mineral resources and SPE-PRMS for energy resources. A detailed mapping between the classifications used by countries and the SEEA-2012 classification into three classes (A, B and C) is available in the following Excel file: http://stats.oecd.org/wbos/fileview2.aspx?IDFile=c7f5f4ad-9b66-4b8c-a8b9-7a6775bd2a28


https://stats.oecd.org/Index.aspx?DataSetCode=NAT_RES

```
data/NAT_RES_27072021094814087.csv
```

**good for availability prediction but limited number of countries**

### 15 - IHS data for IHS - Oil Markets, Midstream & Downstream__29_07_2021 - World

```
'data/IHS - Oil Markets, Midstream & Downstream__29_07_2021 - World.csv'
```

> data provided by Docland from IHS Markit database

### 16 - Paid energy databae

https://www.enerdata.net/research/energy-market-data-co2-emissions-database.html

Very good database it seems, to e evaluated if reserves data are good

