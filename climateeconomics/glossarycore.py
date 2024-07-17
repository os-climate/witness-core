'''
Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from copy import copy, deepcopy

import numpy as np
import pandas as pd

from climateeconomics.database import DatabaseWitnessCore


def get_ref_var_name(var_name: str) -> str:
    return f"{var_name}_ref"


def get_ref_variable(var_name: str, unit: str, default_value=None) -> dict:
    """returns a description for a variable"""
    variable_description = {
        "var_name": var_name,
        "description": f"Normalisation reference for {var_name}",
        "namespace": "ns_ref",
        "type": "float",
        "unit": unit,
    }
    if default_value is not None:
        variable_description.update({"default": default_value})

    return variable_description


class GlossaryCore:
    """Glossary gathering variables used in witness core"""

    # Trillion $ / T$   /   10^12 - Tera works too !
    # Giga$      / G$   /   10^9
    # Million$   / M$   /   10^6
    # Megatons : 1e6 tons
    # Gigatons : 1e9 tons = 1e3 Megatons
    # PWh = 1e3 TWh
    # 1 TWh  = 1e9 kWh = 1e12 Wh

    NB_POLES_COARSE: int = 7  # number of poles in witness coarse
    NB_POLES_UTILIZATION_RATIO = 10  # number of poles for bspline design variables utilization ratio
    Years = "years"
    YearStart = "year_start"
    YearStartDefault = 2020
    YearEnd = "year_end"
    YearEndDefault = 2100
    YearEndDefaultTest = 2030
    YearEndVar = {
        "type": "int",
        "default": YearEndDefault,
        "unit": "year",
        "visibility": "Shared",
        "namespace": "ns_public",
        "range": [2000, 2300],
    }
    TimeStep = "time_step"
    # todo in the futur: merge these 3 invest values
    InvestValue = "invest"
    InvestLevelValue = "invest_level"
    InvestmentsValue = "investment"
    ccus_type = "CCUS"
    CheckRangeBeforeRunBoolName = "check_range_before_run_bool_name"
    SectorGdpPart = "Part of the GDP per sector [T$]"
    ChartSectorGDPPercentage = "Part of the GDP per sector [%]"
    SectionGdpPart = "Part of the GDP per section [T$]"
    ChartSectionGDPPercentage = "Part of the GDP per section [%]"
    SectionEmissionPart = "Part of the total emission per section [GtCO2eq]"
    SectionEmissionPartMt = "Part of the total emission per section [MtCO2eq]"
    SectionEnergyEmissionPart = "Part of the energy emission per section [GtCO2eq]"
    SectionNonEnergyEmissionPart = "Part of the non energy emission per section [GtCO2eq]"
    SectionEnergyConsumptionPart = "Part of the energy consumption per section [PWh]"
    SectionEnergyEmissionPartMt = "Part of the energy emission per section [MtCO2eq]"
    SectionNonEnergyEmissionPartMt = "Part of the non energy emission per section [MtCO2eq]"
    SectionEnergyConsumptionPartTWh = "Part of the energy consumption per section [TWh]"
    EconomicSectors = "Economic sectors"
    Households = "Households"

    AgricultureAndLandUse = "Agriculture & Land Use"
    Energy = "Energy"
    NonEnergy = "Non energy (from economy)"

    ChartGDPPerGroup = "GDP-PPP adjusted per group [T$]"
    ChartPercentagePerGroup = "Percentage per group [%]"
    ChartGDPBiggestEconomies = "Chart of the biggest countries GDP-PPP adjusted per year[G$]"
    ConstraintLowerBoundUsableCapital = "Lower bound usable capital constraint"
    EnergyWasted = "energy wasted [TWh]"
    EnergyWastedObjective = "energy_wasted_objective"
    ConsumptionObjective = "consumption_objective"

    ShareNonEnergyInvestmentsValue = "share_non_energy_investment"
    CO2EmissionsGtValue = "co2_emissions_Gt"
    CO2TaxesValue = "CO2_taxes"
    DamageFractionDfValue = "damage_fraction_df"
    EconomicsDfValue = "economics_df"
    SectorGdpDfValue = "sector_gdp_df"
    SectionGdpDfValue = "section_gdp_df"
    SectionEmissionDfValue = "section_emission_df"
    SectionEnergyEmissionDfValue = "section_energy_emission_df"
    SectionNonEnergyEmissionDfValue = "section_non_energy_emission_df"
    SectionEnergyConsumptionDfValue = "section_energy_consumption_df"
    SectionGdpPercentageDfValue = "section_gdp_percentage_df"
    SectionEnergyConsumptionPercentageDfValue = "section_energy_consumption_percentage_df"
    SectionNonEnergyEmissionGdpDfValue = "section_non_energy_emission_gdp_df"
    SectorEnergyConsumptionPercentageDfName = "sector_emission_consumption_percentage_df"
    PopulationDfValue = "population_df"
    TemperatureDfValue = "temperature_df"
    UtilityDfValue = "utility_df"
    EnergyInvestmentsValue = "energy_investment"
    EnergyInvestmentsWoTaxValue = "energy_investment_wo_tax"
    EnergyInvestmentsWoRenewableValue = "energy_investment_wo_renewable"
    NonEnergyInvestmentsValue = "non_energy_investment"
    EnergyInvestmentsFromTaxValue = "energy_investment_from_tax"  # T$
    WelfareObjective = "welfare_objective"
    NegativeWelfareObjective = "negative_welfare_objective"
    energy_list = "energy_list"
    techno_list = "technologies_list"
    ccs_list = "ccs_list"
    UsableCapitalObjectiveName = "usable_capital_objective"
    UsableCapitalObjectiveRefName = "usable_capital_objective_ref"
    invest_mix = "invest_mix"
    SectorServices = "Services"
    SectorAgriculture = "Agriculture"
    SectorIndustry = "Industry"
    DefaultSectorListGHGEmissions = [SectorServices, SectorIndustry]
    SectorNonEco = "Household"
    SectorEnergy = "energy"
    TotalGDPGroupDFName = "total_gdp_per_group_df"
    PercentageGDPGroupDFName = "percentage_gdp_group_df"
    GDPCountryDFName = "gdp_per_country_df"
    CountryName = "country_name"
    GroupName = "group"
    GDPName = "gdp"
    MeanPercentageName = "mean_percentage"
    TotalEnergyConsumptionSectorName = "total_energy_consumption_sector"
    TotalEnergyConsumptionAllSectorsName = "total_energy_consumption_all_sectors"
    TotalEnergyEmissionsAllSectorsName = "total_energy_emissions_all_sectors"
    TotalNonEnergyEmissionsAllSectorsName = "total_non_energy_emissions_all_sectors"
    TotalEnergyEmissionsSectorName = "total_energy_emissions_sector"
    TotalNonEnergyEmissionsSectorName = "total_non_energy_emissions_sector"
    TotalEmissionsName = "total_emissions"
    ConsumptionObjectiveRefValue = get_ref_var_name(ConsumptionObjective)
    ConsumptionObjectiveRef = get_ref_variable(var_name=ConsumptionObjectiveRefValue, unit="T$", default_value=250)

    # Diet
    Fish = "fish"
    OtherFood = "other"
    FishDailyCal = "fish_calories_per_day"
    OtherDailyCal = "other_calories_per_day"

    TechnoCapitalValue = "techno_capital"
    TechnoConsumptionWithoutRatioValue = "techno_consumption_woratio"
    ConstructionDelay = "construction_delay"

    # namespaces
    NS_MACRO = "ns_macro"
    NS_SECTORS = "ns_sectors"
    NS_WITNESS = "ns_witness"
    NS_ENERGY_MIX = "ns_energy_mix"
    NS_REFERENCE = "ns_ref"
    NS_FUNCTIONS = "ns_functions"
    NS_CCS = "ns_ccs"
    NS_REGIONALIZED_POST_PROC = "ns_regionalized"
    NS_SECTORS_POST_PROC_EMISSIONS = "ns_sectors_postproc"
    NS_SECTORS_POST_PROC_GDP = "ns_sectors_postproc_gdp"
    NS_GHGEMISSIONS = "ns_ghg_emissions"
    NS_HOUSEHOLDS_EMISSIONS = "ns_households_emissions"

    SectionA = "Agriculture, forestry and fishing"
    SectionB = "Mining and quarrying"
    SectionC = "Manufacturing"
    SectionD = "Electricity, gas, steam and air conditioning supply"
    SectionE = "Water supply; sewerage, waste management and remediation activities"
    SectionF = "Construction"
    SectionG = "Wholesale and retail trade; repair of motor vehicles and motorcycles"
    SectionH = "Transportation and storage"
    SectionI = "Accommodation and food service activities"
    SectionJ = "Information and communication"
    SectionK = "Financial and insurance activities"
    SectionL = "Real estate activities"
    SectionM = "Professional, scientific and technical activities"
    SectionN = "Administrative and support service activities"
    SectionO = "Public administration and defence; compulsory social security"
    SectionP = "Education"
    SectionQ = "Human health and social work activities"
    SectionR = "Arts, entertainment and recreation"
    SectionS = "Other service activities"
    SectionT = "Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use"
    SectionU = "Activities of extraterritorial organizations and bodies"
    SectionHousehold = "Household"

    SectionsAgriculture = [SectionA]
    SectionsIndustry = [SectionB, SectionC, SectionD, SectionE, SectionF]
    SectionsServices = [
        SectionG,
        SectionH,
        SectionI,
        SectionJ,
        SectionK,
        SectionL,
        SectionM,
        SectionN,
        SectionO,
        SectionP,
        SectionQ,
        SectionR,
        SectionS,
        SectionT,
    ]
    SectionsNonEco = [SectionHousehold]

    SectionsPossibleValues = [
        SectionA,
        SectionB,
        SectionC,
        SectionD,
        SectionE,
        SectionF,
        SectionG,
        SectionH,
        SectionI,
        SectionJ,
        SectionK,
        SectionL,
        SectionM,
        SectionN,
        SectionO,
        SectionP,
        SectionQ,
        SectionR,
        SectionS,
        SectionT,
    ]

    SectionDictSectors = {
        SectorAgriculture: SectionsAgriculture,
        SectorIndustry: SectionsIndustry,
        SectorServices: SectionsServices,
    }

    SectionListValue = "section_list"

    SectionList = {
        "var_name": SectionListValue,
        "type": "list",
        "description": "List of sub-sectors",
        "subtype_descriptor": {"list": "string"},
        "default": SectionsPossibleValues,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "editable": False,
        "structuring": True,
    }

    SectionGdpPercentageDf = {
        "var_name": SectionGdpPercentageDfValue,
        "type": "dataframe",
        "unit": "%",
        "description": "Percentage of the GDP for each sub-sector",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {Years: ("int", [1900, YearEndDefault], False)},
    }
    SectionEnergyConsumptionPercentageDf = {
        "var_name": SectionEnergyConsumptionPercentageDfValue,
        "type": "dataframe",
        "unit": "%",
        "description": "Percentage of the energy consumption for each sub-sector",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {Years: ("int", [1900, YearEndDefault], False)},
    }
    SectorEnergyConsumptionPercentageDf = {
        "var_name": SectorEnergyConsumptionPercentageDfName,
        "type": "dataframe",
        "unit": "%",
        "description": "Percentage of the energy consumption for each sector",
        "dynamic_dataframe_columns": True,
        "default": DatabaseWitnessCore.EnergyConsumptionPercentageSectorDict.value,
    }

    SectionNonEnergyEmissionGdpDf = {
        "var_name": SectionNonEnergyEmissionGdpDfValue,
        "type": "dataframe",
        "unit": "tCO2eq/Million $GDP",
        "description": "Non energy CO2 emission per $GDP",
        "dynamic_dataframe_columns": True,
    }

    SectionNonEnergyEmissionGdpDfSector = {
        "var_name": SectionNonEnergyEmissionGdpDfValue,
        "type": "dataframe",
        "unit": "tCO2eq/Million $GDP",
        "description": "Non energy CO2 emission per $GDP per section",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {Years: ("int", [1900, YearEndDefault], False)},
    }

    SectionNonEnergyEmissionGDPDfSector = {
        "type": "dataframe",
        "unit": "tCO2eq/Million $GDP",
        "description": "Non energy CO2 emission per $GDP",
        "dynamic_dataframe_columns": True,
    }

    SectorsPossibleValues = [
        SectorServices,
        SectorAgriculture,
        SectorIndustry,
    ]
    SectorListValue = "sector_list"

    SectorList = {
        "var_name": SectorListValue,
        "type": "list",
        "description": "List of sectors",
        "subtype_descriptor": {"list": "string"},
        "default": SectorsPossibleValues,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "editable": False,
        "structuring": True,
    }

    MissingSectorNameValue = "sector_name_deduced_share"
    MissingSectorName = {
        "var_name": MissingSectorNameValue,
        "type": "string",
        "default": SectorsPossibleValues[-1],
        "editable": False,
        "structuring": True,
    }

    CaloriesPerCapitaValue = "calories_pc_df"
    CaloriesPerCapita = {
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "kcal/day/person",
        "dataframe_descriptor": {
            Years: ("float", None, True),
            "kcal_pc": ("float", None, True),
        },
    }

    CaloriesPerCapitaBreakdownValue = "calories_pc_breakdown_df"
    CaloriesPerCapitaBreakdown = {
        "var_name": CaloriesPerCapitaBreakdownValue,
        "type": "dataframe",
        "unit": "kcal/day/person",
    }

    CarbonCycleDfValue = "carboncycle_df"
    CarbonCycleDf = {
        "var_name": CarbonCycleDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            "atmo_conc": ("float", [0, 1e30], False),
            "lower_ocean_conc": ("float", [0, 1e30], False),
            "shallow_ocean_conc": ("float", [0, 1e30], False),
            "ppm": ("float", [0, 1e30], False),
            "atmo_share_since1850": ("float", [0, 1e30], False),
            "atmo_share_sinceystart": ("float", [0, 1e30], False),
        },
    }

    CO2DamagePrice = "CO2_damage_price"
    CO2DamagePriceDf = {
        "var_name": CO2DamagePrice,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "$/tCO2Eq",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            CO2DamagePrice: ("float", [0, 1e30], False),
        },
    }
    CO2DamagePriceInitValue = "init_CO2_damage_price"
    CO2DamagePriceInitVar = {
        "varname": CO2DamagePriceInitValue,
        "type": "float",
        "default": 25.0,
        "unit": "$/tCO2Eq",
        "user_level": 2,
    }

    ExtraCO2tDamagePrice = "Extra tCO2Eq damage price"
    ExtraCO2tDamagePriceDf = {
        "var_name": ExtraCO2tDamagePrice,
        "type": "dataframe",
        "unit": "$/tCO2Eq",
        "description": "Damage of an extra (wrt pre-industria levels) ton of CO2 equivalent"
        " in the atmosphere on the economy",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            ExtraCO2tDamagePrice: ("float", [0, 1e30], False),
        },
    }

    CO2EmissionsDetailDfValue = "CO2_emissions_detail_df"
    CO2EmissionsDfValue = "CO2_emissions_df"
    CO2EmissionsDetailDf = {
        "var_name": CO2EmissionsDetailDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "Gt",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            "sigma": ("float", [0, 1e30], False),
            "gr_sigma": ("float", None, False),
            "land_emissions": ("float", [0, 1e30], False),
            "cum_land_emissions": ("float", [0, 1e30], False),
            "indus_emissions": ("float", [0, 1e30], False),
            "cum_indus_emissions": ("float", [0, 1e30], False),
            "total_emissions": ("float", [0, 1e30], False),
            "cum_total_emissions": ("float", [0, 1e30], False),
        },
    }

    CO2EmissionsDf = {
        "var_name": CO2EmissionsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "Gt",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            "total_emissions": ("float", [-1.0e9, 1.0e9], False),
            "cum_total_emissions": ("float", [-1.0e9, 1.0e9], False),
        },
    }

    TotalCO2Emissions = "Total CO2 emissions"
    CO2EmissionsGt = {
        "var_name": CO2EmissionsGtValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_ENERGY_MIX,
        "unit": "Gt",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            TotalCO2Emissions: ("float", None, False),
        },
    }
    CO2TaxEfficiencyValue = "CO2_tax_efficiency"
    CO2TaxEfficiency = {
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            CO2TaxEfficiencyValue: ("float", [0.0, 100.0], False),
        },
    }

    CO2Tax = "CO2_tax"
    CO2Taxes = {
        "var_name": CO2TaxesValue,
        "type": "dataframe",
        "unit": "$/tCO2Eq",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            CO2Tax: ("float", None, True),
        },
        "dataframe_edition_locked": False,
    }

    ExtraCO2EqSincePreIndustrialValue = "Extra CO2Eq since pre-industrial era"
    ExtraCO2EqSincePreIndustrialDf = {
        "var_name": ExtraCO2EqSincePreIndustrialValue,
        "type": "dataframe",
        "description": "Extra gigatons of CO2 Equivalent in the atmosphere with respect to pre-industrial level. "
        "For GHG other than CO2, the conversion is done on a 20 year basis.",
        "namespace": NS_WITNESS,
        "visibility": "Shared",
        "unit": "GtCO2Eq",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            ExtraCO2EqSincePreIndustrialValue: ("float", [0, 1e30], False),
        },
    }
    ExtraCO2EqSincePreIndustrialDetailedValue = f"{ExtraCO2EqSincePreIndustrialValue} (detailed)"
    ExtraCO2EqSincePreIndustrial2OYbasisValue = f"{ExtraCO2EqSincePreIndustrialValue} (20-year basis)"
    ExtraCO2EqSincePreIndustrial10OYbasisValue = f"{ExtraCO2EqSincePreIndustrialValue} (100-year basis)"
    ExtraCO2EqSincePreIndustrialDetailedDf = {
        "var_name": ExtraCO2EqSincePreIndustrialDetailedValue,
        "type": "dataframe",
        "description": "Extra gigatons of CO2 Equivalent in the atmosphere with respect to pre-industrial level.",
        "namespace": NS_WITNESS,
        "visibility": "Shared",
        "unit": "GtCO2Eq",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            ExtraCO2EqSincePreIndustrial2OYbasisValue: ("float", [0, 1e30], False),
            ExtraCO2EqSincePreIndustrial10OYbasisValue: ("float", [0, 1e30], False),
        },
    }

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    GreenHouseGases = [CO2, CH4, N2O]
    YearBasis20 = "(20-year basis)"
    YearBasis100 = "(100-year basis)"
    GlobalWarmingPotentialdDfValue = "Global warming potential"
    GlobalWarmingPotentialdDf = {
        "var_name": GlobalWarmingPotentialdDfValue,
        "type": "dataframe",
        "description": "Global warming potential in gigatons of  CO2 Eq",
        "unit": "GtCO2Eq",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            f"{CO2} {YearBasis20}": ("float", [0, 1e30], False),
            f"{CH4} {YearBasis20}": ("float", [0, 1e30], False),
            f"{N2O} {YearBasis20}": ("float", [0, 1e30], False),
            f"{CO2} {YearBasis100}": ("float", [0, 1e30], False),
            f"{CH4} {YearBasis100}": ("float", [0, 1e30], False),
            f"{N2O} {YearBasis100}": ("float", [0, 1e30], False),
        },
    }

    DietMortalityParamDf = {
        "var_name": "diet_mortality_param_df",
        "type": "dataframe",
        "default": "default_diet_mortality_param_df",
        "user_level": 3,
        "unit": "-",
        "dataframe_descriptor": {
            "param": ("string", None, False),
            "undernutrition": ("float", [0, 1e30], True),
            "overnutrition": ("float", [0, 1e30], True),
        },
    }

    Alpha = "alpha"
    DamageToProductivity = "damage_to_productivity"
    DamageFractionOutput = "damage_frac_output"
    BaseCarbonPrice = "base_carbon_price"
    DamageFractionDf = {
        "var_name": DamageFractionDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            DamageFractionOutput: ("float", [0.0, 1.0], False),
        },
    }
    Damages = "Damages [G$]"
    DamageDfValue = "damage_df"
    DamagesFromClimate = "Damages from climate [G$]"
    DamagesFromProductivityLoss = "Damages from productivity loss [G$]"
    EstimatedDamages = "Estimated damages [G$]"
    DamageDf = {
        "var_name": DamageDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            Damages: ("float", [0, 1e30], False),
            EstimatedDamages: ("float", [0, 1e30], False),
        },
    }

    EstimatedDamagesFromProductivityLoss = "Estimated damages from productivity loss (not applied) [G$]"
    EstimatedDamagesFromClimate = "Estimated damages from climate (not applied) [G$]"
    DamageDetailedDfValue = "damage_detailed_df"
    DamageDetailedDf = {
        "var_name": DamageDetailedDfValue,
        "type": "dataframe",
        "namespace": NS_MACRO,
        "visibility": "Shared",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            Damages: ("float", [0, 1e30], False),  # G$
            DamagesFromClimate: ("float", [0, 1e30], False),  # G$
            DamagesFromProductivityLoss: ("float", [0, 1e30], False),  # G$
            EstimatedDamages: ("float", [0, 1e30], False),  # G$
            EstimatedDamagesFromClimate: ("float", [0, 1e30], False),  # G$
            EstimatedDamagesFromProductivityLoss: ("float", [0, 1e30], False),  # G$
        },
    }

    InitialGrossOutput = {
        "var_name": "init_gross_output",
        "type": "float",
        "unit": "G$",
        "visibility": "Shared",
        "default": 130.187,
        "namespace": NS_WITNESS,
        "user_level": 2,
    }

    Output = "output"  # todo in the future: delete this key, it corresponds to gross output
    GrossOutput = "gross_output"  # trillion $
    NetOutput = "net_output"  # todo in the future: delete this key, it corresponds to gross output net of damage,
    OutputNetOfDamage = "output_net_of_d"  # trillion $
    Consumption = "consumption"
    PerCapitaConsumption = "pc_consumption"

    # The number of columns depends dynamically on SectorsList
    SectorGdpDf = {
        "var_name": SectorGdpDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
        },
    }

    # The number of columns depends dynamically on SectionsList
    SectionGdpDf = {
        "var_name": SectionGdpDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_GHGEMISSIONS,
        "description": "GDP values of sub-sectors in a sector",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
        },
    }

    TotalEmissions = "Total emissions"

    EmissionsDfValue = "emissions_df"
    EconomicsEmissionDfValue = "economics_emissions_df"
    EnergyEmissions = "Energy emissions"
    NonEnergyEmissions = "Non energy emissions"
    insertGHGNonEnergyEmissions = "Non energy {} emissions of economy"
    insertGHGTotalEmissions = "Total {} emissions"
    insertGHGEnergyEmissions = f"{Energy}" + " {} emissions"
    insertGHGAgriLandEmissions = f"{AgricultureAndLandUse}" + " {} emissions"
    EmissionDf = {
        "type": "dataframe",
        "description": "Emissions of macroeconomics (all sectors)",
        "unit": "GtCO2eq",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            TotalEmissions: ("float", [0, 1e30], False),
            EnergyEmissions: ("float", [0, 1e30], False),
            NonEnergyEmissions: ("float", [0, 1e30], False),
        },
    }

    SectionEmissionDf = {
        "var_name": SectionEmissionDfValue,
        "type": "dataframe",
        "description": "Total emission per section",
        "unit": "GtCO2eq",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
        },
    }

    SectionEnergyEmissionDf = {
        "var_name": SectionEnergyEmissionDfValue,
        "type": "dataframe",
        "description": "Energy emission per section",
        "unit": "GtCO2eq",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
        },
    }

    SectionNonEnergyEmissionDf = {
        "var_name": SectionNonEnergyEmissionDfValue,
        "type": "dataframe",
        "description": "Non-energy emission per section",
        "unit": "GtCO2eq",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
        },
    }

    SectionEnergyConsumptionDf = {
        "var_name": SectionEnergyConsumptionDfValue,
        "type": "dataframe",
        "description": "Energy consumption per section",
        "unit": "PWh",
        "visibility": "Shared",
        "namespace": NS_GHGEMISSIONS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
        },
    }

    AllSectionsGdpDfValue = "all_sections_gdp_df"
    AllSectionsGdpDf = {
        "var_name": AllSectionsGdpDfValue,
        "type": "dataframe",
        "description": "GDP of all sections",
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
        },
    }

    EconomicsDf = {
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            GrossOutput: ("float", [0, 1e30], False),
            OutputNetOfDamage: ("float", [0, 1e30], False),
            PerCapitaConsumption: ("float", [0, 1e30], False),
        },
    }

    EconomicsDetailDfValue = "economics_detail_df"
    Productivity = "productivity"
    ProductivityWithDamage = "Productivity with damages"
    ProductivityWithoutDamage = "Productivity without damages"
    ProductivityGrowthRate = "productivity_gr"
    OutputGrowth = "output_growth"
    OptimalEnergyProduction = "Optimal Energy Production [TWh]"
    UsedEnergy = "Used Energy [TWh]"
    UnusedEnergy = "Unused Energy [TWh]"
    EnergyUsage = "Energy Usage"
    EconomicsDetailDf = {
        "var_name": EconomicsDetailDfValue,
        "type": "dataframe",
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            GrossOutput: ("float", [0, 1e30], False),  # G$
            OutputNetOfDamage: ("float", [0, 1e30], False),  # G$
            Productivity: ("float", [0, 1e30], False),
            ProductivityWithDamage: ("float", [0, 1e30], False),
            ProductivityWithoutDamage: ("float", [0, 1e30], False),
            ProductivityGrowthRate: ("float", None, False),
            Consumption: ("float", [0, 1e30], False),  # G$
            PerCapitaConsumption: ("float", [0, 1e30], False),
            InvestmentsValue: ("float", [0, 1e30], False),  # G$
            EnergyInvestmentsValue: ("float", [0, 1e30], False),  # G$
            EnergyInvestmentsWoTaxValue: ("float", [0, 1e30], False),  # G$
            NonEnergyInvestmentsValue: ("float", [0, 1e30], False),  # G$
            EnergyInvestmentsFromTaxValue: ("float", None, False),  # T$
            OutputGrowth: ("float", None, False),
            UsedEnergy: ("float", [0, 1e30], False),
            UnusedEnergy: ("float", [0, 1e30], False),
            OptimalEnergyProduction: ("float", [0, 1e30], False),
            EnergyWasted: ("float", [0, 1e30], False),
        },
    }
    PopulationValue = "population"
    PopulationDf = {
        "var_name": PopulationDfValue,
        "type": "dataframe",
        "unit": "millions of people",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            PopulationValue: ("float", None, False),
        },
    }

    EnergyMeanPriceValue = "energy_mean_price"

    EnergyMeanPriceObjectiveValue = f"{EnergyMeanPriceValue}_objective"
    EnergyMeanPriceObjective = {
        "var_name": EnergyMeanPriceObjectiveValue,
        "type": "array",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
        "unit": "-",
    }
    EnergyMeanPriceObjectiveRefValue = get_ref_var_name(EnergyMeanPriceObjectiveValue)
    EnergyMeanPriceObjectiveRef = get_ref_variable(
        var_name=EnergyMeanPriceObjectiveRefValue,
        unit="$",
        default_value=100.0,
    )

    StreamPricesValue = "energy_prices"  # todo : rename streams_prices, but it will break all l1
    ResourcesPriceValue = "resources_price"

    ResourcesPrice = {
        "type": "dataframe",
        "unit": "$/t",
        "visibility": "Shared",
        "namespace": "ns_resource",
    }


    ResourcesCO2Emissions = {
        "type": "dataframe",
        "unit": "kgCO2/kg",
        "visibility": "Shared",
        "namespace": "ns_resource",
    }

    EnergyPriceValue = "energy_price"
    EnergyMeanPrice = {
        "var_name": EnergyMeanPriceValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_ENERGY_MIX,
        "unit": "$/MWh",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            EnergyPriceValue: ("float", [0, 1e30], True),
        },
    }

    EnergyProductionValue = "energy_production"
    EnergyProductionDetailedValue = "energy_production_detailed"
    EnergyProcductionWithoutRatioValue = "energy_production_woratio"
    EnergyConsumptionValue = "energy_consumption"
    EnergyConsumptionWithoutRatioValue = "energy_consumption_woratio"
    LandUseRequiredValue = "land_use_required"

    TotalProductionValue = "Total production"
    EnergyProductionDf = {
        "var_name": EnergyProductionValue,
        "type": "dataframe",
        "visibility": "Shared",
        "unit": "PWh",
        "namespace": NS_ENERGY_MIX,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            TotalProductionValue: ("float", [0, 1e30], False),
        },
    }

    EnergyProductionDetailedDf = {
        "var_name": EnergyProductionValue,
        "type": "dataframe",
        "unit": "TWh",
        "dynamic_dataframe_columns": True,
    }

    EnergyCarbonIntensityDfValue = "energy_carbon_intensity_df"
    EnergyCarbonIntensityDf = {
        "var_name": EnergyCarbonIntensityDfValue,
        "type": "dataframe",
        "unit": "kgCO2Eq/kWh",
        "description": "Total CO2 equivalent emitted by energy mix divided by total energy production of energy mix",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            EnergyCarbonIntensityDfValue: ("float", [0, 1e30], False),
        },
    }

    EnergyProductionDfSectors = {
        "var_name": EnergyProductionValue,
        "type": "dataframe",
        "visibility": "Shared",
        "unit": "PWh",
        "namespace": NS_SECTORS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            TotalProductionValue: ("float", [0, 1e30], False),
        },
    }

    EnergyProductionResidentialValue = "energy_residential_production"
    EnergyProductionDfResidential = {
        "var_name": EnergyProductionResidentialValue,
        "type": "dataframe",
        "visibility": "Shared",
        "description": "Energy production dedicated to residential",
        "unit": "PWh",
        "namespace": NS_SECTORS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            TotalProductionValue: ("float", [0, 1e30], False),
        },
    }

    EnergyInvestments = {
        "var_name": EnergyInvestmentsValue,
        "type": "dataframe",
        "unit": "100G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            EnergyInvestmentsValue: ("float", [0.0, 1e30], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
    }

    EnergyInvestmentsMinimizationObjective = "Energy invest minimization objective"
    EnergyInvestmentsWoTax = {  # output of IndependentInvestDiscipline & input of MacroeconomicsDiscipline
        "var_name": EnergyInvestmentsWoTaxValue,
        "type": "dataframe",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            EnergyInvestmentsWoTaxValue: ("float", [0.0, 1e30], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
    }

    FoodWastePercentageValue = "food_waste_percentage"
    FoodWastePercentage = {
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            FoodWastePercentageValue: ("float", [0.0, 100.0], False),
        },
    }

    OrganicWasteValue = "organic_waste"
    OrganicWaste = {
        "type": "dataframe",
        "unit": "kcal",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            "kcal": ("float", [0.0, 1e6], False),
        },
    }

    GHGEmissionsDetailedDfValue = "GHG_emissions_detail_df"
    GHGEmissionsDfValue = "GHG_emissions_df"
    TotalN2OEmissions = f"Total {N2O} emissions"
    TotalCH4Emissions = f"Total {CH4} emissions"
    GHGEmissionsDf = {
        "var_name": GHGEmissionsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "Gt",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            TotalCO2Emissions: ("float", [0, 1e30], False),
            TotalN2OEmissions: ("float", [0, 1e30], False),
            TotalCH4Emissions: ("float", [0, 1e30], False),
        },
    }

    GWPEmissionsDfValue = "GWP_emissions"
    TotalGWPEmissionsDfValue = "Total GWP emissions"
    GWPEmissionsDf = {
        "type": "dataframe",
        "description": f"Data on Global warming potential for the three main green house gases : {GreenHouseGases}",
        "unit": "GtCO2Eq",
    }

    GHGCycleDfValue = "ghg_cycle_df"
    CO2Concentration = f"{CO2} (ppm)"
    CH4Concentration = f"{CH4} (ppb)"
    N2OConcentration = f"{N2O} (ppb)"
    GHGCycleDf = {
        "varname": GHGCycleDfValue,
        "type": "dataframe",
        "description": f"Concentrations forecasts of the three main green house gases : {GreenHouseGases}",
        "unit": "ppm",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            CO2Concentration: ("float", [0.0, 1e6], True),
            CH4Concentration: ("float", [0.0, 1e6], True),
            N2OConcentration: ("float", [0.0, 1e6], True),
        },
        "visibility": "Shared",
        "namespace": NS_WITNESS,
    }

    RenewablesEnergyInvestmentsValue = "Renewables energy investments [100G$]"
    RenewablesEnergyInvestments = {
        "var_name": RenewablesEnergyInvestmentsValue,
        "namespace": NS_WITNESS,
        "type": "dataframe",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            InvestmentsValue: ("float", [0.0, 1e30], True),
        },
        "unit": "100G$",
    }

    EnergyInvestmentsWoRenewable = {
        "var_name": EnergyInvestmentsWoRenewableValue,
        "type": "dataframe",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            EnergyInvestmentsWoRenewableValue: ("float", [0.0, 1e30], True),
        },
        "unit": "100G$",
    }

    ShareNonEnergyInvestment = {
        "var_name": ShareNonEnergyInvestmentsValue,
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            ShareNonEnergyInvestmentsValue: ("float", [0.0, 100.0], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
    }

    ExoGForcing = "exog_forcing"
    Forcing = "forcing"
    TempAtmo = "temp_atmo"
    TempOcean = "temp_ocean"
    TemperatureDf = {
        "var_name": TemperatureDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "°C",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            TempAtmo: ("float", None, False),
        },
    }

    TemperatureDetailedDfValue = "temperature_detailed_df"
    TemperatureDetailedDf = {
        "var_name": TemperatureDetailedDfValue,
        "type": "dataframe",
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            ExoGForcing: ("float", None, False),
            Forcing: ("float", None, False),
            TempAtmo: ("float", None, False),
            TempOcean: ("float", None, False),
        },
    }

    UtilityDiscountRate = "u_discount_rate"
    DiscountedQuantityUtilityPopulation = "Discounted quantity utility population"
    PerCapitaUtilityQuantity = "Utility quantity per capita"
    DiscountedUtilityQuantityPerCapita = "Discounted utility quantity per capita"
    PeriodUtilityPerCapita = "period_utility_pc"
    DiscountedUtility = "discounted_utility"
    Welfare = "welfare"
    UtilityDf = {
        "var_name": UtilityDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            UtilityDiscountRate: ("float", [0, 100], False),
            PerCapitaUtilityQuantity: ("float", None, False),
            DiscountedUtilityQuantityPerCapita: ("float", None, False),
            DiscountedQuantityUtilityPopulation: ("float", None, False),
        },
        "unit": "-",
    }

    QuantityObjectiveValue = "Quantity_objective"
    LastYearUtilityObjectiveValue = "last_year_utility_obj"

    QuantityObjective = {
        "var_name": QuantityObjectiveValue,
        "type": "array",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
        "description": "objective of quantity of things consumed. Quantity  = Consumption / Price",
        "unit": "-",
    }

    LastYearUtilityObjective = {
        "var_name": LastYearUtilityObjectiveValue,
        "type": "array",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
        "description": "utility of last year",
        "unit": "-",
    }

    DecreasingGdpIncrementsObjectiveValue = "decreasing_gdp_increments_obj"

    DecreasingGdpIncrementsObjective = {
        "var_name": DecreasingGdpIncrementsObjectiveValue,
        "type": "array",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
        "description": "Here to minimize areas where net gpp is decreasing. Objective should be minimized. "
        "Self normalized, no need for reference division.",
        "unit": "-",
    }

    NetGdpGrowthRateObjectiveValue = "net_gdp_growth_rate_obj"

    NetGdpGrowthRateObjective = {
        "var_name": NetGdpGrowthRateObjectiveValue,
        "type": "array",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
        "description": "Net Gdp growth rate obj",
        "unit": "-",
    }

    ProductionDfValue = "production_df"
    ProductionDf = {
        "var_name": ProductionDfValue,
        "namespace": NS_SECTORS,
        "visibility": "Shared",
        "type": "dataframe",
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            GrossOutput: ("float", [0, 1e30], False),
            OutputNetOfDamage: ("float", [0, 1e30], False),
        },
    }

    CapitalDfValue = "capital_df"
    Capital = "capital"
    UsableCapital = "usable_capital"
    UsableCapitalUnbounded = "Unbounded usable capital [G$]"
    NonEnergyCapital = "non_energy_capital"
    CapitalDf = {
        "var_name": CapitalDfValue,
        "namespace": NS_WITNESS,
        "visibility": "Shared",
        "type": "dataframe",
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            Capital: ("float", [0, 1e30], False),
            UsableCapital: ("float", [0, 1e30], False),
        },
    }

    EnergyCapitalDfValue = "energy_capital"
    EnergyCapitalDf = {
        "var_name": EnergyCapitalDfValue,
        "type": "dataframe",
        "unit": "G$",
        "description": "Capital of energy in G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            Capital: ("float", [0, 1e30], False),
        },
        "visibility": "Shared",
        "namespace": NS_WITNESS,
    }

    DetailedCapitalDfValue = "detailed_capital_df"
    Emax = "e_max"
    EnergyEfficiency = "energy_efficiency"
    DetailedCapitalDf = {
        "var_name": DetailedCapitalDfValue,
        "visibility": "Shared",
        "namespace": NS_MACRO,
        "type": "dataframe",
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            Capital: ("float", [0, 1e30], False),
            NonEnergyCapital: ("float", [0, 1e30], False),
            UsableCapital: ("float", [0, 1e30], False),
            Emax: ("float", [0, 1e30], False),
            EnergyEfficiency: ("float", [0, 1e30], False),
        },
    }

    SectorizedEconomicsDf = {  # todo: miss per capita consumption !
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            GrossOutput: ("float", [0, 1e30], False),
            OutputNetOfDamage: ("float", [0, 1e30], False),
            Capital: ("float", [0, 1e30], False),
        },
    }

    SectorizedEconomicsDetailDf = {  # todo: miss per capita consumption !
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            GrossOutput: ("float", [0, 1e30], False),
            OutputNetOfDamage: ("float", [0, 1e30], False),
            Capital: ("float", [0, 1e30], False),
            UsableCapital: ("float", [0, 1e30], False),
            OutputGrowth: ("float", None, False),
            Damages: ("float", [0, 1e30], False),
            Consumption: ("float", [0, 1e30], False),
        },
    }

    ProductivityDfValue = "productivity_df"
    ProductivityDf = {
        "var_name": ProductivityDfValue,
        "type": "dataframe",
        "unit": "-",
        "description": "productivity levels through years, applied, with damage, and without wamage.",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            Productivity: ("float", [0, 1e30], False),
            ProductivityGrowthRate: ("float", None, False),
            ProductivityWithoutDamage: ("float", [0, 1e30], False),
            ProductivityWithDamage: ("float", [0, 1e30], False),
            OptimalEnergyProduction: ("float", [0, 1e30], False),
            UsedEnergy: ("float", [0, 1e30], False),
            UnusedEnergy: ("float", [0, 1e30], False),
            EnergyWasted: ("float", [0, 1e30], False),
        },
    }

    AllSectorsDemandDfValue = "all_sector_demand_df"
    AllSectorsDemandDf = {
        "var_name": AllSectorsDemandDfValue,
        "type": "dataframe",
        "unit": "T$",
        "description": "all sectors demands aggregated",
        "dataframe_descriptor": {},
        "dynamic_dataframe_columns": True,
    }

    RedistributionInvestmentsDfValue = "redistribution_investments_df"
    RedistributionInvestmentsDf = {
        "var_name": RedistributionInvestmentsDfValue,
        "type": "dataframe",
        "unit": "G$",
        "dataframe_descriptor": {},
        "dynamic_dataframe_columns": True,
    }

    RedistributionEnergyProductionDfValue = "redistribution_energy_production_df"
    RedistributionEnergyProductionDf = {
        "var_name": RedistributionEnergyProductionDfValue,
        "type": "dataframe",
        "unit": "PWh",
        "dataframe_descriptor": {},
        "dynamic_dataframe_columns": True,
    }

    ShareSectorInvestmentDfValue = "share_sector_investment_df"
    ShareInvestment = "Share of total investments [%]"
    ShareSectorInvestmentDf = {
        "type": "dataframe",
        "unit": "%",
        "description": "Amount of output net of damage allocated to the specific sector",
        "visibility": "Shared",
        "namespace": NS_SECTORS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            ShareInvestment: ("float", [0.0, 100.0], False),
        },
    }

    ShareSectorEnergyDfValue = "share_sector_energy_df"
    ShareSectorEnergy = "Share of total energy production [%]"
    ShareSectorEnergyDf = {
        "type": "dataframe",
        "unit": "%",
        "description": "Amount of the total energy production attributed to the specific sector",
        "visibility": "Shared",
        "namespace": NS_SECTORS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            ShareSectorEnergy: ("float", [0.0, 100.0], False),
        },
    }
    ResidentialCategory = "Residential"
    ShareResidentialEnergyDfValue = "share_residential_energy_df"
    ShareResidentialEnergyDf = {
        "type": "dataframe",
        "unit": "%",
        "description": "Amount of the total energy production attributed to residential",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            ShareSectorEnergy: ("float", [0.0, 100.0], False),
        },
    }
    ResidentialEnergyConsumptionDfValue = "residential_energy_production_df"
    ResidentialEnergyConsumptionDf = {
        "var_name": RedistributionEnergyProductionDfValue,
        "type": "dataframe",
        "unit": "PWh",
        "description": "Energy that is consumed by residential sector",
        "visibility": "Shared",
        "namespace": NS_GHGEMISSIONS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            TotalProductionValue: ("float", None, False),
        },
    }

    ResidentialEmissionsDfValue = "residential_emissions_df"
    ResidentialEmissionsDf = {
        "type": "dataframe",
        "unit": "Gt CO2 Eq",
        "description": "Emission by residential sector (only due to energy consumption)",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            TotalEmissions: ("float", None, False),
        },
    }

    OtherEnergyCategory = "Other"
    ShareOtherEnergyDfValue = "share_other_energy_df"
    ShareOtherEnergyDf = {
        "type": "dataframe",
        "unit": "%",
        "description": "Amount of the total energy production attributed to other category",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            ShareSectorEnergy: ("float", [0.0, 100.0], False),
        },
    }

    FractionDamageToProductivityValue = "frac_damage_prod"
    FractionDamageToProductivity = {
        "var_name": FractionDamageToProductivityValue,
        "type": "float",
        "default": 0.3,
        "user_level": 2,
        "unit": "-",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
    }

    PandemicParamDfValue = "pandemic_param_df"
    PandemicParamDf = {
        "var_name": PandemicParamDfValue,
        "type": "dataframe",
        "default": DatabaseWitnessCore.PandemicParamsDf.value,
        "unit": "-",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            "param": ("string", None, False),
            "disability": ("float", [0, 1e30], True),
            "mortality": ("float", [0, 1e30], True),
        },
    }

    WorkforceDfValue = "workforce_df"
    EmploymentRate = "employment_rate"
    Workforce = "workforce"
    WorkforceDf = {
        "var_name": WorkforceDfValue,
        "type": "dataframe",
        "unit": "millions of people",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {},
        "dynamic_dataframe_columns": True,
    }
    WorkingAgePopulationDfValue = "working_age_population_df"
    Population1570 = "population_1570"
    WorkingAgePopulationDf = {
        "var_name": WorkingAgePopulationDfValue,
        "type": "dataframe",
        "unit": "millions of people",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            Population1570: ("float", [0, 1e30], False),
        },
    }

    InvestmentDfValue = "investment_df"
    InvestmentDf = {
        "var_name": InvestmentDfValue,
        "type": "dataframe",
        "unit": "G$",
        "visibility": "Shared",
        "namespace": NS_SECTORS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            InvestmentsValue: ("float", [0, 1e30], False),
        },
    }

    SectorDemandPerCapitaDfValue = "sector_demand_per_capita"
    SectorDemandPerCapitaDf = {
        "var_name": SectorDemandPerCapitaDfValue,
        "type": "dataframe",
        "unit": "$/person",
        "visibility": "Shared",
        "namespace": NS_SECTORS,
        "description": "Sector demand per person per year [$/year]",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            SectorDemandPerCapitaDfValue: ("float", [0, 1e30], False),
        },
    }

    SectorGDPDemandDfValue = "GDP sector demand [G$]"
    SectorGDPDemandDf = {
        "var_name": SectorGDPDemandDfValue,
        "type": "dataframe",
        "unit": "T$",
        "visibility": "Shared",
        "namespace": NS_SECTORS,
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            SectorGDPDemandDfValue: ("float", [0, 1e30], False),
        },
    }

    InvestmentShareGDPValue = "total_investment_share_of_gdp"
    InvestmentShareGDP = {
        "var_name": InvestmentShareGDPValue,
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "share_investment": ("float", [0.0, 100.0], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
    }

    InvestmentBeforeYearStartValue = "invest_before_ystart"
    InvestmentBeforeYearStartDf = {
        "var_name": InvestmentBeforeYearStartValue,
        "type": "dataframe",
        "unit": "G$",
        "dataframe_descriptor": {
            "past years": ("int", [-20, -1], True),
            "invest": ("float", [0, 1e30], True),
        },
        "dataframe_edition_locked": False,
    }

    ShareMaxInvestName = "share_max_invest"
    ShareMaxInvest = {
        "var_name": ShareMaxInvestName,
        "type": "float",
        "unit": "%",
        "default": 10.0,
        "description": "float to set maximum percentage of GDP to allow to investments in sectors and energy",
    }

    UtilisationRatioValue = "Utilisation Ratio [%]"

    MaxInvestConstraintName = "max_invest_constraint"
    MaxInvestConstraint = {
        "var_name": MaxInvestConstraintName,
        "type": "array",
        "unit": "[]",
        "description": "Max investment in sectors constraint using share_max_invest percentage",
    }

    MaxInvestConstraintRefName = "max_invest_constraint_ref"
    MaxInvestConstraintRef = {
        "var_name": MaxInvestConstraintRefName,
        "type": "float",
        "unit": "G$",
        "default": 100.0,
        "user_level": 3,
        "description": "Max investment reference to normalize associated constraint",
    }

    MaxBudgetValue = "Max budget"
    MaxBudgetConstraintValue = "Max budget constraint"
    MaxBudgetDf = {
        "var_name": MaxBudgetValue,
        "type": "dataframe",
        "description": "Maximum budget that can be invested in Energy production and CCUS technos",
        "unit": "G$",
        "visibility": "Shared",
        "namespace": NS_ENERGY_MIX,
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            MaxBudgetValue: ("float", [0.0, 1e12], True),
        },
    }

    MaxBudgetConstraint = {
        "var_name": MaxBudgetConstraintValue,
        "type": "array",
        "description": "Maximum budget that can be invested in Energy production and CCUS technos",
        "unit": "G$",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
    }

    MaxBudgetConstraintRefValue = get_ref_var_name(MaxBudgetConstraintValue)
    MaxBudgetConstraintRef = get_ref_variable(var_name=MaxBudgetConstraintRefValue, unit="T$", default_value=1e4)

    UsableCapitalObjective = {
        "var_name": UsableCapitalObjectiveName,
        "type": "array",
        "unit": "-",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
        "description": "Usable capital objective",
    }

    UsableCapitalObjectiveRef = {
        "var_name": UsableCapitalObjectiveRefName,
        "type": "float",
        "unit": "T$",
        "default": 100.0,
        "user_level": 3,
        "visibility": "Shared",
        "namespace": NS_REFERENCE,
        "description": "reference to normalize usable capital objective",
    }

    TargetEnergyProductionValue = "Target energy production"
    TargetProductionConstraintValue = "Target production constraint"
    TargetEnergyProductionDf = {
        "var_name": TargetEnergyProductionValue,
        "type": "dataframe",
        "description": " Energy Production",
        "unit": "TWh",
        "visibility": "Shared",
        "namespace": NS_ENERGY_MIX,
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            TargetEnergyProductionValue: ("float", [0.0, 1e30], True),
        },
    }

    TargetProductionConstraint = {
        "var_name": TargetProductionConstraintValue,
        "type": "array",
        "description": "Production Constraint",
        "unit": "TWh",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
    }

    TargetProductionConstraintRefValue = get_ref_var_name(TargetProductionConstraintValue)
    TargetProductionConstraintRef = get_ref_variable(
        var_name=TargetProductionConstraintRefValue, unit="TWh", default_value=1e5
    )

    CheckRangeBeforeRunBool = {
        "var_name": CheckRangeBeforeRunBoolName,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "type": "bool",
        "default": False,
    }

    # objective functions
    CO2EmissionsObjectiveValue = "CO2EmissionsObjective"
    CO2EmissionsObjective = {
        "var_name": CO2EmissionsObjectiveValue,
        "type": "array",
        "unit": "-",
        "visibility": "Shared",
        "namespace": NS_FUNCTIONS,
        "description": "Objective on Total CO2 emissions, mean of emissions between 2020 and 2100. Can be negative",
    }

    CO2EmissionsRef = {
        "var_name": "CO2EmissionsRef",
        "type": "float",
        "default": DatabaseWitnessCore.CumulativeCO2Emissions.value / (2022 - 1750 + 1.0),
        "unit": "Gt",
        "visibility": "Shared",
        "namespace": NS_REFERENCE,
        "description": "Mean CO2 emissions produced from fossil fuels and industry between 1750 and 2022",
    }

    StreamsCO2EmissionsValue = "energy_CO2_emissions"  # todo : rename streams_co2_emissions, but it will break all l1
    StreamsCO2Emissions = {
        "var_name": StreamsCO2EmissionsValue,
        "type": "dataframe",
        "unit": "kg/kWh ... to be checked for CCUS streams", # fixme todo
        "visibility": "Shared",
        "namespace": NS_ENERGY_MIX,
        "dynamic_dataframe_columns": True,
    }

    TotalEnergyEmissions = "Total Energy emissions"
    TotalEnergyCO2eqEmissionsDf = {
        "var_name": TotalEnergyEmissions,
        "type": "dataframe",
        "unit": "GtCO2Eq",
        "dataframe_descriptor": {
            Years: ("float", [1900, YearEndDefault], False),
            TotalEnergyEmissions: ("float", [0.0, 1e30], True),
        },
    }

    TotalGDPGroupDF = {
        "var_name": TotalGDPGroupDFName,
        "type": "dataframe",
        "unit": "T$",
    }
    PercentageGDPGroupDF = {
        "var_name": PercentageGDPGroupDFName,
        "type": "dataframe",
        "unit": "%",
    }
    GDPCountryDF = {
        "var_name": GDPCountryDFName,
        "type": "dataframe",
        "unit": "G$",
    }

    TempOutput = "TempOutput"
    TempOutputDf = {
        "var_name": TempOutput,
        "type": "dataframe",
        "description": "used to debug some gradients",
        "dataframe_descriptor": {
            Years: ("int", [1900, YearEndDefault], False),
            NonEnergyCapital: ("int", [1900, YearEndDefault], False),
        },
    }

    @staticmethod
    def get_dynamic_variable(variable: dict):
        """to be used with dynamic inputs/outputs"""
        return copy(variable)

    @staticmethod
    def delete_namespace(variable: dict):
        # todo : doesnt work
        """delete the namespace of variable"""
        out = deepcopy(variable)
        try:
            del out["namespace"]
        except KeyError:
            pass
        return out

    @staticmethod
    def set_namespace(variable: dict, namespace: str):
        # todo : doesnt work
        """set the namespace for a variable"""
        out = deepcopy(variable)
        out["namespace"] = namespace
        return out

    @staticmethod
    def get_random_dataframe(years, df_variable, min_val: float = 0.0, max_val: float = 100.0):
        out = {}
        for key in df_variable["dataframe_descriptor"].keys():
            if key == GlossaryCore.Years:
                out[key] = years
            else:
                out[key] = np.random.uniform(min_val, max_val)
        return pd.DataFrame(out)

    @staticmethod
    def get_random_dataframe_columns(years, columns: list[str], min_val: float = 0.0, max_val: float = 100.0):
        out = {GlossaryCore.Years: years}
        for key in columns:
            out[key] = np.random.uniform(min_val, max_val)
        return pd.DataFrame(out)
