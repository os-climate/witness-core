
# Documentation for Agriculture economic sector Discipline

This document provides an overview of the Agriculture Economy discipline. This discipline mainly computes the GDP and damages, and the price of each food product, based on productions coming from the Crop discipline.

The discipline handles a list of food types which currenyly includes:
- Red meat (representing beef, buffalo, lamb, mutton, sheep, goat, horse)
- White meat (representing poultry, pig, turkey, rabbit)
- Fish
- Milk
- Eggs
- Rice
- Maize
- Other cereals (than rice and maize)
- Sugar cane
- Fruits and vegetables
- Other (what remains and is not modeled explicitly)

**This function computes the economic and damage metrics for a given food type based on input parameters and data. The results are aggregated across all food types to model the overall economic and damage impact on the agriculture sector. Below is an explanation of the code, the calculations, and the outputs.**

---

## Inputs

| **Input**                                        | **Description**                                                                       |
|--------------------------------------------------|---------------------------------------------------------------------------------------|
| **`capital_food_type`**: `np.ndarray`           | Represents the capital allocated to this specific food type.                          |
| **`production_loss_from_prod_loss`**: `np.ndarray` | Represents production losses due to productivity declines (in metric tons, Mt).       |
| **`production_loss_from_immediate_climate_damages`**: `np.ndarray` | Represents production losses directly caused by climate damages (in metric tons, Mt). |
| **`production_delivered_to_consumers`**: `np.ndarray` | Represents the amount of production reaching consumers (in metric tons, Mt).          |
| **`production_for_all_streams`**: `np.ndarray`  | Represents the total production allocated across all streams (in metric tons, Mt).    |
| **`energy_price`**: `np.ndarray`                | Represents the energy price (in \$/MWh).                                              |
| **`params`**: `dict`                            | A dictionary containing various parameters, including:                                |
|                                                  | - Labor intensity per unit of production (`$/ton`).                                   |
|                                                  | - Energy intensity per unit of production (`kWh/ton`).                                |
|                                                  | - Capital maintenance and amortization costs.                                         |
|                                                  | - Feeding costs (`$/ton`).                                                            |
|                                                  | - Fertilization and pesticides (`$/ton`).                                             |
|                                                  | - Margin share of the final price.                                                    |

  

---

## Outputs

1. **`outputs`**: `dict`  
   Contains various economic and damage metrics, detailed in the *Output Details* section.

2. **`price_breakdown_df`**: `pd.DataFrame`  
   A detailed breakdown of the final price components for the given food type.

---

## Calculations for one food type

The results are then aggregated on all food types to computes Agriculture GDP, net GDP, damages.

### 1. **Unitary Price (`final_price`)**
The final price per kilogram is computed by summing up various cost components and applying a profit margin:

- **Labor Cost**: 
$$\text{labor} = \frac{\text{Labor Intensity (in \$/ton)}}{1000}$$

- **Energy Cost**: 
$$\text{energy} = \frac{\text{Energy Intensity (in kWh/ton)} \times \text{Energy Price (in \$/MWh)}}{1e6}$$

- **Capital Costs**:
  - **Maintenance Cost**: Taken directly from `params`.
  - **Amortization Cost**: Taken directly from `params`.

- **Feeding Costs**:
$$\text{feeding} = \frac{\text{Feeding Costs (in \$/ton)}}{1000}$$

- **Fertilization and pesticides**:
$$\text{fertilization\ and\ pesticides} = \frac{\text{Fertilization and pesticides Costs (in \$/ton)}}{1000}$$

- **Price without margin**:
$$\text{price\ without\ margin} = \text{labor} + \text{energy} + \text{feeding} + \text{fertilization\ and\ pesticides} + \text{capital\ maintenance} + \text{capital\ amortization}$$
- **Margin**:  
  The profit margin is applied as a fraction of the pre-margin price:
$$\text{margin} = \frac{\text{Margin Share (\%)} \times \text{Price Without Margin}}{100 - \text{Margin Share (\%)}}$$
> This way, a margin set to 20% leads to the margin represented 20% of the final price

- **Final Price**:
$$\text{final\ price} = \text{price\ wo\ margin} + \text{margin}$$

---

### 2. **Damages**
- **Damages from Productivity Losses**:
$$\text{damages\ prod\ loss} = \frac{\text{Production Loss (in Mt)} \times \text{Final Price (in \$/kg)}}{1000}$$

- **Damages from Immediate Climate Events**:
$$\text{damages\ immediate\ climate\ damages} = \frac{\text{Immediate Loss (in Mt)} \times \text{Final Price (in \$/kg)}}{1000}$$

- **Total Damages**:
$$\text{damages} = \text{damages\ prod\ loss} + \text{damages\ immediate\ climate\ damages}$$

---

### 3. **GDP Contributions**
- **Net GDP from Energy**:
$$\text{net\ gdp\ energy} = \frac{\text{Total Production for All Streams (in Mt)} \times \text{Final Price (in \$/kg)}}{1000}$$

- **Net GDP from Food**:
$$\text{net\ gdp\ food} = \frac{\text{Production Delivered to Consumers (in Mt)} \times \text{Final Price (in \$/kg)}}{1000}$$

- **Total Net GDP**:
$$\text{net\ gdp} = \text{net\ gdp\ food} + \text{net\ gdp\ energy}$$

---

### 4. **Gross Output**
The total gross output is the sum of the net GDP and damages:
$$\text{gross\ output} = \text{net\ gdp} + \text{damages}$$

---

## Output Details

The `outputs` dictionary contains the following keys:

1. **`Damages`**: Total damages (`T$`).
2. **`DamagesFromClimate`**: Damages caused by immediate climate impacts (`T$`).
3. **`DamagesFromProductivityLoss`**: Damages due to productivity losses (`T$`).
4. **`GrossOutput`**: Total gross output (`T$`).
5. **`OutputNetOfDamage`**: Net output after subtracting damages (`T$`).
6. **`CropFoodNetGdp`**: Net GDP contribution from food production (`T$`).
7. **`CropEnergyNetGdp`**: Net GDP contribution from energy production (`T$`).
8. **`FoodTypesPrice`**: Final price of the food type (`$/kg`).

The `price_breakdown_df` DataFrame includes columns for each cost component (`Labor`, `Energy`, etc.) across simulation years.

