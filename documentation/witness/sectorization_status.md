# Status on Sectorization

## Table of Contents

1. [Sectorized Consumption by Sector](#1-sectorized-consumption-by-sector)
   1. [Formula (E1)](#11-formula-e1)
   2. [Reminder: Formula (E2)](#12-reminder-formula-e2)
2. [In `SectorsEnergyRedistributionDiscipline`](#2-in-sectorsenergyredistributiondiscipline)
3. [In `DemandDiscipline` (to be renamed `ConsumptionDiscipline`)](#3-in-demanddiscipline-to-be-renamed-consumptiondiscipline)
   1. [Create the New Outputs](#31-create-the-new-outputs)
   2. [Compute Gradients](#32-compute-gradients)
   3. [Graphs](#33-graphs)
4. [In `ConsumptionDiscipline` (to be renamed `SectorizedUtilityDiscipline`)](#4-in-consumptiondiscipline-to-be-renamed-sectorizedutilitydiscipline)
   1. [Add Inputs](#41-add-inputs)
   2. [Add Output](#42-add-output)
   3. [Graph](#43-graph)
   4. [Tasks](#44-tasks)
5. [Optimization Process for Sectorization](#5-optimization-process-for-sectorization)
   1. [Design Variables](#51-design-variables)
   2. [Objective](#52-objective)
6. [Storytelling Use Cases](#6-storytelling-use-cases)
   1. [Mono Scenarios](#61-mono-scenarios)
   2. [Multi-Scenario with Scenarios 1-4](#62-multi-scenario-with-scenarios-1-4)

---

## 1. Sectorized Consumption by Sector

We need to define sectorized consumption by sector as follows:

### 1.1. Formula (E1)
$$
\text{Consumption sector S} = \text{net GDP sector S} - \text{Invest in sector S} - \left(\text{share energy conso sector S} \times \text{Invest energy}\right)
$$

### 1.2. Reminder: Formula (E2)
$$
\text{Sectorized welfare} = \sum_{\text{sectors } S} w_S \underbrace{\left[ \text{saturation effect of sector S} \left(\frac{\text{conso sector S}}{\text{conso sector S year start}}\right) \right]}_{\text{Utility of sector S}}
$$

- The saturation effect function follows an S-curve.


## 2. In `SectorsEnergyRedistributionDiscipline`

Create the output:
- `all_sectors_share_consos_df` (columns: years, Services, Industry, Agriculture) in namespace `NS_SECTORS`.
  - Contains the share of energy consumed by each sector.


## 3. In `DemandDiscipline` (to be renamed `ConsumptionDiscipline`)

This is where we will compute the consumption for each sector. Add the following inputs:
- `all_sectors_share_consos_df` from `SectorsEnergyRedistributionDiscipline`
- `f'{sector}.{GlossaryCore.InvestmentDfValue}'` for each sector (already existing)
- `f'{sector}.{GlossaryCore.ProductionDfValue}'` for each sector (already existing)
- `GlossaryCore.EnergyInvestmentsWoTaxValue`, as in `MacroeconomicsDiscipline` (sectorized, already existing)

This will allow the computation of formula (E1) in the model.

### 3.1. Create the New Outputs:
- `sectorized_consumption_df` (columns: years, Services, Industry, Agriculture) in namespace `NS_SECTORS`
- `consumption_df` (local variable): (years, Consumption) -> sum of all consumption of the sectors in `NS_SECTORS`.

### 3.2. Compute Gradients

### 3.3. Graphs:
1. One, non-sectorized, showing total GDP breakdown (damage + consumption + invest in sectors + invests in energy).
2. Sectorized graphs, one per sector, showing for each sector the breakdown (damage + consumption + invest in selected sector + share consumption selected sector * invest in energy).
3. Optional: A graph with three lines (one per sector) showing:
   - $$\frac{\text{conso sector S}}{\text{invest sector S} + \text{share energy conso S} \times \text{Invest in energy}}$$
   to see which sector has the highest return on investments.



## 4. In `ConsumptionDiscipline` (to be renamed `SectorizedUtilityDiscipline`)

### 4.1. Add Inputs:
- `sectorized_consumption_df` (mentioned above) in namespace `NS_SECTORS`
- Saturation effect parameters inputs for each sector (dynamically), as in `UtilityDiscipline` (not sectorized).

### 4.2. Add Outputs:
- `{sector}.{utility_obj}` at namespace `NS_FUNCTIONS`, for each sector

> Global welfare objective (E2) will be recomposed in `FuncManagerDiscipline`, to easily tune the weights.

### 4.3. Graph:
For each sector:
- Show S-curve fitting.
- Show consumption variation since year start (%).

### 4.4. Tasks:
- Tune the S-curve parameters for each sector to ensure they make sense.
- Compute gradients of the objective with respect to coupled inputs.



## 5. Optimization Process for Sectorization

### 5.1. Design Variables:
- Invests in technologies (as in non-sectorized version).
- `f'{sector}.{GlossaryCore.ShareSectorInvestmentDfValue}'` for each sector:
  - This gives: $$\text{invests sector S} = \text{design var share invest sector S} \times Q_{\text{tot}}$$
- `f'{sector}.{GlossaryCore.ShareSectorEnergyDfValue}'` for each sector:
  - Controls the distribution of energy.

### 5.2. Objective:
- Sectorized welfare.
- Optionally: Activate for each sector the `EnergyWastedObjective`.


## 6. Storytelling Use Cases:

As in non-sectorized storytelling, plus all sectorized design variables always on in each scenario.

### 6.1. Mono Scenarios:
1. Fossil only, no damage.
2. Fossil only, with damage.
3. Fossil and renewable, no CCUS.
4. All technologies.

### 6.2. Multi-Scenario with Scenarios 1-4.
