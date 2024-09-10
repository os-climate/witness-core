## Labor market model

The goal of the labor market model is to compute the workforce available for each economics sector. The list of economics sectors is:
- agriculture
- industry
- services

### Main Inputs
- sector list: list of economics sectors
-Working age population ($working\_age\_population\_df$): Dataframe with working age population per year in million of people.
- Workforce share per sector ($workforce\_share\_per\_sector$): Dataframe with workforce distribution per sector in %.

### Outputs
- Workforce df ($workforce\_df$): Dataframe with workforce per sector and total workforce in million of people.
- Employment df ($employment\_df$): Dataframe with employment rate per year.

### Time Step
The time step $t$ in each equation represents the period we are looking at. In the inputs we initialize the data with 2020 information.

### Labor force
To obtain the labor force we use the population in working age and the employment rate. We defined the population in working age as the population in the 15-70 age range.
$$L = working\_age\_pop * employment\_rate$$
The employment rate is for now fixed at  $65.9\%$ following International Labour Organization data[^9]. However to take into account the impact of COVID-19 crisis on employment rate, the value is different for 2020-2031 year interval. We used ILO forecast values for 2021, 2022 and 2023 to extrapolate a recovery function until fixed value is reached.
| ![employmentrate.PNG](employmentrate.PNG) |
|:--:|
| *International Labour Organization Employment rate. Copyright International Labor Organization* |
*Note*: The ILO value of employment rate is different from ours because on the graph the employment rate of the total population is represented (number of people employed/ total population) and we use the employment rate for 15-70 population (number of employed people/ working_age_pop)

### Labor force per sector
For each economics sector we compute the workforce using the total workforce and the distribution per sector (in inputs).
The workforce for a sector s is then:
$$L_s = L * sector\_share\_workforce$$

### Other inputs
-  Employment rate recovery function parameters: $employment\_a\_param$, $employment\_power\_param$, $employment\_rate\_base\_value$

## References
[^9]: International Labour Organization, ILOSTAT database. Data retrieved on February 8, 2022.
