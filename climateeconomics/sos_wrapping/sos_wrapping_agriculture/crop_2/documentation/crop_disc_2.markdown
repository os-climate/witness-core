# Crop Model Documentation

## Overview
The `compute_food_type` function in the `crop_2.py` file is responsible for computing various metrics for a specific food type. This function takes into account several inputs related to investments, energy consumption, workforce, productivity, emissions, and more. The outputs of this function provide detailed information on land use, greenhouse gas emissions, food waste, and other relevant data.

## Outputs and Formulas

### Land Use for Food Production (`land_use_food`)
**Formula:**
\[ \text{land\_use\_food} = \text{production\_raw} \times \left(1 - \frac{\text{share\_dedicated\_to\_biomass\_dry\_prod}}{100} + \frac{\text{share\_dedicated\_to\_biomass\_wet\_prod}}{100}\right) \times \frac{\text{land\_use\_by\_prod\_unit}}{10^5} \]
**Explanation:**
- `production_raw`: Raw production based on investment and capital expenditure.
- `share_dedicated_to_biomass_dry_prod`: Share of production dedicated to dry biomass.
- `share_dedicated_to_biomass_wet_prod`: Share of production dedicated to wet biomass.
- `land_use_by_prod_unit`: Land use required per unit of the food type.

### Food Waste Before Distribution (`food_waste_before_distribution`)
**Formula:**
\[ \text{food\_waste\_before\_distribution} = \text{production\_for\_consumers} \times \frac{\text{share\_food\_waste\_before\_distribution}}{100} \]
**Explanation:**
- `production_for_consumers`: Net production available for consumers after accounting for biomass production.
- `share_food_waste_before_distribution`: Share of food waste occurring before distribution.

### Food Waste by Consumers (`food_waste_by_consumers`)
**Formula:**
\[ \text{food\_waste\_by\_consumers} = \text{production\_delivered\_to\_consumers} \times \frac{\text{share\_food\_waste\_by\_consumers}}{100} \]
**Explanation:**
- `production_delivered_to_consumers`: Production delivered to consumers after accounting for food waste before distribution.
- `share_food_waste_by_consumers`: Share of food waste occurring by consumers.

### Production Delivered to Consumers (`production_delivered_to_consumers`)
**Formula:**
\[ \text{production\_delivered\_to\_consumers} = \text{production\_for\_consumers} - \text{food\_waste\_before\_distribution} \]
**Explanation:**
- `production_for_consumers`: Net production available for consumers after accounting for biomass production.
- `food_waste_before_distribution`: Food waste occurring before distribution.

### CO2 Emissions from Food Production (`co2_emissions_food`)
**Formula:**
\[ \text{co2\_emissions\_food} = \text{production\_before\_waste} \times \text{co2\_emissions\_per\_prod\_unit} \times \left(1 - \frac{\text{share\_dedicated\_to\_biomass\_dry\_prod}}{100} - \frac{\text{share\_dedicated\_to\_biomass\_wet\_prod}}{100}\right) \]
**Explanation:**
- `production_before_waste`: Production before accounting for waste.
- `co2_emissions_per_prod_unit`: CO2 emissions produced per unit of the food type.
- `share_dedicated_to_biomass_dry_prod`: Share of production dedicated to dry biomass.
- `share_dedicated_to_biomass_wet_prod`: Share of production dedicated to wet biomass.

### CH4 Emissions from Food Production (`ch4_emissions_food`)
**Formula:**
\[ \text{ch4\_emissions\_food} = \text{production\_before\_waste} \times \text{ch4\_emissions\_per\_prod\_unit} \times \left(1 - \frac{\text{share\_dedicated\_to\_biomass\_dry\_prod}}{100} - \frac{\text{share\_dedicated\_to\_biomass\_wet\_prod}}{100}\right) \]
**Explanation:**
- `production_before_waste`: Production before accounting for waste.
- `ch4_emissions_per_prod_unit`: CH4 emissions produced per unit of the food type.
- `share_dedicated_to_biomass_dry_prod`: Share of production dedicated to dry biomass.
- `share_dedicated_to_biomass_wet_prod`: Share of production dedicated to wet biomass.

### N2O Emissions from Food Production (`n2o_emissions_food`)
**Formula:**
\[ \text{n2o\_emissions\_food} = \text{production\_before\_waste} \times \text{n2o\_emissions\_per\_prod\_unit} \times \left(1 - \frac{\text{share\_dedicated\_to\_biomass\_dry\_prod}}{100} - \frac{\text{share\_dedicated\_to\_biomass\_wet\_prod}}{100}\right) \]
**Explanation:**
- `production_before_waste`: Production before accounting for waste.
- `n2o_emissions_per_prod_unit`: N2O emissions produced per unit of the food type.
- `share_dedicated_to_biomass_dry_prod`: Share of production dedicated to dry biomass.
- `share_dedicated_to_biomass_wet_prod`: Share of production dedicated to wet biomass.

### Production Wasted Due to Productivity Loss (`production_wasted_by_productivity_loss`)
**Formula:**
\[ \text{production\_wasted\_by\_productivity\_loss} = \text{production\_raw} \times \frac{\text{crop\_productivity\_reduction}}{100} \]
**Explanation:**
- `production_raw`: Raw production based on investment and capital expenditure.
- `crop_productivity_reduction`: Reduction in crop productivity.

### Production Wasted Due to Immediate Damages (`production_wasted_by_immediate_damages`)
**Formula:**
\[ \text{production\_wasted\_by\_immediate\_damages} = \text{production\_wasted\_by\_productivity\_loss} \times \text{damage\_fraction} \]
**Explanation:**
- `production_wasted_by_productivity_loss`: Production wasted due to productivity loss.
- `damage_fraction`: Fraction of production lost due to damages.

### Production Dedicated to Dry Biomass (`production_dedicated_to_biomass_dry`)
**Formula:**
\[ \text{production\_dedicated\_to\_biomass\_dry} = \text{net\_production} \times \frac{\text{share\_dedicated\_to\_biomass\_dry\_prod}}{100} \]
**Explanation:**
- `net_production`: Net production after accounting for productivity loss and immediate damages.
- `share_dedicated_to_biomass_dry_prod`: Share of production dedicated to dry biomass.

### Production Dedicated to Wet Biomass (`production_dedicated_to_biomass_wet`)
**Formula:**
\[ \text{production\_dedicated\_to\_biomass\_wet} = \text{net\_production} \times \frac{\text{share\_dedicated\_to\_biomass\_wet\_prod}}{100} \]
**Explanation:**
- `net_production`: Net production after accounting for productivity loss and immediate damages.
- `share_dedicated_to_biomass_wet_prod`: Share of production dedicated to wet biomass.

### Total Dry Biomass Production Available (`total_biomass_dry_prod_available`)
**Formula:**
\[ \text{total\_biomass\_dry\_prod\_available} = \text{production\_dedicated\_to\_biomass\_dry} + \text{food\_waste\_before\_distribution\_reused\_for\_energy\_prod\_biomass\_dry} + \text{consumers\_waste\_reused\_for\_energy\_prod\_biomass\_dry} \]
**Explanation:**
- `production_dedicated_to_biomass_dry`: Production dedicated to dry biomass.
- `food_waste_before_distribution_reused_for_energy_prod_biomass_dry`: Food waste before distribution reused for dry biomass energy production.
- `consumers_waste_reused_for_energy_prod_biomass_dry`: Consumers' waste reused for dry biomass energy production.

### Total Wet Biomass Production Available (`total_biomass_wet_prod_available`)
**Formula:**
\[ \text{total\_biomass\_wet\_prod\_available} = \text{production\_dedicated\_to\_biomass\_wet} + \text{food\_waste\_before\_distribution\_reused\_for\_energy\_prod\_biomass\_wet} + \text{consumers\_waste\_reused\_for\_energy\_prod\_biomass\_wet} \]
**Explanation:**
- `production_dedicated_to_biomass_wet`: Production dedicated to wet biomass.
- `food_waste_before_distribution_reused_for_energy_prod_biomass_wet`: Food waste before distribution reused for wet biomass energy production.
- `consumers_waste_reused_for_energy_prod_biomass_wet`: Consumers' waste reused for wet biomass energy production.

### Calories Produced for Consumers (`kcal_produced_for_consumers`)
**Formula:**
\[ \text{kcal\_produced\_for\_consumers} = \text{production\_delivered\_to\_consumers} \times \text{kcal\_per\_prod\_unit} \]
**Explanation:**
- `production_delivered_to_consumers`: Production delivered to consumers after accounting for food waste before distribution.
- `kcal_per_prod_unit`: Calories provided per unit of the food type.

### Calories per Person per Day (`kcal_per_pers_per_day`)
**Formula:**
\[ \text{kcal\_per\_pers\_per\_day} = \frac{\text{kcal\_produced\_for\_consumers}}{\text{population} \times 365} \times 1000 \]
**Explanation:**
- `kcal_produced_for_consumers`: Total calories produced for consumers.
- `population`: Population data used for calculations.

## Computation Steps
1. **Investment, Energy, and Workforce Allocation**: Calculate the investment, energy, and workforce allocated to the specific food type.
2. **Raw Production Calculation**: Compute the raw production based on investment and capital expenditure.
3. **Limiting Ratio Calculation**: Determine the limiting ratio based on energy and workforce needs and availability.
4. **Production Adjustments**: Adjust the raw production based on the limiting ratio, productivity loss, and immediate damages.
5. **Emissions Calculation**: Compute the CO2, CH4, and N2O emissions based on production before waste and emissions per unit.
6. **Net Production Calculation**: Calculate the net production after accounting for productivity loss and immediate damages.
7. **Biomass Production Calculation**: Determine the production dedicated to dry and wet biomass.
8. **Food Waste Calculation**: Compute the food waste before distribution and by consumers.
9. **Energy Production from Waste**: Calculate the food waste reused for energy production.
10. **Total Biomass Production Calculation**: Compute the total biomass production available for energy.
11. **Calorie Calculation**: Determine the calories produced for consumers and the daily calories per person.
12. **Land Use Calculation**: Compute the land use for food production based on raw production and land use per unit.