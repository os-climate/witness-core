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
from copy import deepcopy, copy


class GlossaryCore:
    """Glossary gathering variables used in witness core"""

    Years = "years"
    YearStart = "year_start"
    YearEnd = "year_end"
    TimeStep = "time_step"
    # todo in the futur: merge these 3 invest values
    InvestValue = "invest"
    InvestLevelValue = "invest_level"
    InvestmentsValue = "investment"

    SectorGdpPart = "Part of the GDP per sector [T$]"
    ChartSectorGDPPercentage = "Part of the GDP per sector [%]"
    SectionGdpPart = "Part of the GDP per section [T$]"
    ChartSectionGDPPercentage = "Part of the GDP per section [%]"

    ConstraintLowerBoundUsableCapital = "Lower bound usable capital constraint"
    EnergyWasted = "energy wasted [TWh]"
    EnergyWastedObjective = "energy_wasted_objective"
    ShareNonEnergyInvestmentsValue = "share_non_energy_investment"
    CO2EmissionsGtValue = "co2_emissions_Gt"
    CO2TaxesValue = "CO2_taxes"
    DamageFractionDfValue = "damage_fraction_df"
    EconomicsDfValue = "economics_df"
    SectorGdpDfValue = "sector_gdp_df"
    SectionGdpDfValue = "section_gdp_df"
    SectionGdpDictValue = "detailed_section_gdp"
    SectionGdpPercentageDfValue = "section_gdp_percentage_df"
    PopulationDfValue = "population_df"
    TemperatureDfValue = "temperature_df"
    UtilityDfValue = "utility_df"
    EnergyInvestmentsValue = "energy_investment"
    EnergyInvestmentsWoTaxValue = "energy_investment_wo_tax"
    EnergyInvestmentsWoRenewableValue = "energy_investment_wo_renewable"
    NonEnergyInvestmentsValue = "non_energy_investment"
    EnergyInvestmentsFromTaxValue = "energy_investment_from_tax"  # G$
    WelfareObjective = "welfare_objective"
    NormalizedWelfare = "Normalized welfare"
    NegativeWelfareObjective = "negative_welfare_objective"
    LastYearDiscountedUtilityObjective = "last_year_discounted_utility_objective"
    energy_list = "energy_list"
    techno_list = "technologies_list"
    ccs_list = "ccs_list"

    invest_mix = "invest_mix"
    SectorServices = "Services"
    SectorAgriculture = "Agriculture"
    SectorIndustry = "Industry"
    SectorEnergy = "energy"

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

    SectionsAgriculture = [SectionA]
    SectionsIndustry = [SectionB, SectionC, SectionD, SectionE, SectionF]
    SectionsServices = [SectionG, SectionH, SectionI, SectionJ, SectionK, SectionL, SectionM, SectionN, SectionO, SectionP, SectionQ, SectionR, SectionS, SectionT]

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
    SectionListValue = "section_list"

    SectionList = {
        "var_name": SectionListValue,
        "type": "list",
        "subtype_descriptor": {"list": "string"},
        "default": SectionsPossibleValues,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "editable": False,
        "structuring": True,
    }
    df_descriptor_section_df = {section: ('float', [0., 100.], True) for section in SectionsPossibleValues}
    df_descriptor_section_df.update({Years: ("int", [1900, 2100], False)})
    SectionGdpPercentageDf = {
        "var_name": SectionGdpPercentageDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": df_descriptor_section_df,
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

    CarbonCycleDfValue = "carboncycle_df"
    CarbonCycleDf = {
        "var_name": CarbonCycleDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            "atmo_conc": ("float", None, False),
            "lower_ocean_conc": ("float", None, False),
            "shallow_ocean_conc": ("float", None, False),
            "ppm": ("float", None, False),
            "atmo_share_since1850": ("float", None, False),
            "atmo_share_sinceystart": ("float", None, False),
        },
    }

    CO2DamagePrice = "CO2_damage_price"
    CO2DamagePriceDf = {
        "var_name": CO2DamagePrice,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "$/tCO2",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            CO2DamagePrice: ("float", None, False),
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
            Years: ("float", None, False),
            "sigma": ("float", None, False),
            "gr_sigma": ("float", None, False),
            "land_emissions": ("float", None, False),
            "cum_land_emissions": ("float", None, False),
            "indus_emissions": ("float", None, False),
            "cum_indus_emissions": ("float", None, False),
            "total_emissions": ("float", None, False),
            "cum_total_emissions": ("float", None, False),
        },
    }

    CO2EmissionsDf = {
        "var_name": CO2EmissionsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "Gt",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "total_emissions": ("float", None, False),
            "cum_total_emissions": ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            TotalCO2Emissions: ("float", None, False),
        },
    }
    CO2TaxEfficiencyValue = "CO2_tax_efficiency"
    CO2TaxEfficiency = {
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            CO2TaxEfficiencyValue: ("float", None, False),
        },
    }

    CO2Tax = "CO2_tax"
    CO2Taxes = {
        "var_name": CO2TaxesValue,
        "type": "dataframe",
        "unit": "$/tCO2",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            CO2Tax: ("float", None, True),
        },
        "dataframe_edition_locked": False,
    }

    DietMortalityParamDf = {
        "var_name": "diet_mortality_param_df",
        "type": "dataframe",
        "default": "default_diet_mortality_param_df",
        "user_level": 3,
        "unit": "-",
        "dataframe_descriptor": {
            "param": ("string", None, False),
            "undernutrition": ("float", None, True),
            "overnutrition": ("float", None, True),
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
            Years: ("int", [1900, 2100], False),
            DamageFractionOutput: ("float", [0., 1.], False),
            BaseCarbonPrice: ("float", None, False),
        },
    }
    Damages = "Damages [G$]"
    DamageDfValue = "damage_df"
    DamagesFromClimate = "Damages from climate [G$]"
    DamagesFromProductivityLoss = "Damages from productivity loss [G$]"
    DamageDf = {
        "var_name": DamageDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            Damages: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            Damages: ("float", None, False),  # G$
            DamagesFromClimate: ("float", None, False),  # G$
            DamagesFromProductivityLoss: ("float", None, False),  # G$
            EstimatedDamagesFromClimate: ("float", None, False),  # G$
            EstimatedDamagesFromProductivityLoss: ("float", None, False),  # G$
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

    Output = (
        "output"  # todo in the future: delete this key, it corresponds to gross output
    )
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
            Years: ("int", [1900, 2100], False),
        },
    }

    # The number of columns depends dynamically on SectionsList
    SectionGdpDf = {
        "var_name": SectionGdpDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
        },
    }

    # The number of columns depends dynamically on SectionsList
    SectionGdpDict = {
        "var_name": SectionGdpDictValue,
        "type": "dict",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "G$",
    }

    EconomicsDf = {
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            GrossOutput: ("float", None, False),
            OutputNetOfDamage: ("float", None, False),
            PerCapitaConsumption: ("float", None, False),
            EnergyWasted: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            GrossOutput: ("float", None, False),  # G$
            OutputNetOfDamage: ("float", None, False),  # G$
            Productivity: ("float", None, False),
            ProductivityWithDamage: ("float", None, False),
            ProductivityWithoutDamage: ("float", None, False),
            ProductivityGrowthRate: ("float", None, False),
            Consumption: ("float", None, False),  # G$
            PerCapitaConsumption: ("float", None, False),
            InvestmentsValue: ("float", None, False),  # G$
            EnergyInvestmentsValue: ("float", None, False),  # G$
            EnergyInvestmentsWoTaxValue: ("float", None, False),  # G$
            NonEnergyInvestmentsValue: ("float", None, False),  # G$
            EnergyInvestmentsFromTaxValue: ("float", None, False),  # G$
            OutputGrowth: ("float", None, False),
            UsedEnergy: ("float", None, False),
            UnusedEnergy: ("float", None, False),
            OptimalEnergyProduction: ("float", None, False),
            EnergyWasted: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            PopulationValue: ("float", None, False),
        },
    }

    EnergyMeanPriceValue = "energy_mean_price"

    EnergyPricesValue = "energy_prices"
    ResourcesPriceValue = "resources_price"
    EnergyPriceValue = "energy_price"
    EnergyMeanPrice = {
        "var_name": EnergyMeanPriceValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_ENERGY_MIX,
        "unit": "$/MWh",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            EnergyPriceValue: ("float", None, True),
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
            Years: ("int", [1900, 2100], False),
            TotalProductionValue: ("float", None, False),
        },
    }

    EnergyProductionDfSectors = {
        "var_name": EnergyProductionValue,
        "type": "dataframe",
        "visibility": "Shared",
        "unit": "PWh",
        "namespace": NS_SECTORS,
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            TotalProductionValue: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            TotalProductionValue: ("float", None, False),
        },
    }

    EnergyInvestments = {
        "var_name": EnergyInvestmentsValue,
        "type": "dataframe",
        "unit": "100G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            EnergyInvestmentsValue: ("float", [0.0, 1e30], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": NS_WITNESS,
    }

    EnergyInvestmentsMinimizationObjective = "Energy invest minimization objective"
    EnergyInvestmentsWoTax = (
        {  # output of IndependentInvestDiscipline & input of MacroeconomicsDiscipline
            "var_name": EnergyInvestmentsWoTaxValue,
            "type": "dataframe",
            "unit": "G$",
            "dataframe_descriptor": {
                Years: ("int", [1900, 2100], False),
                EnergyInvestmentsWoTaxValue: ("float", [0.0, 1e30], True),
            },
            "dataframe_edition_locked": False,
            "visibility": "Shared",
            "namespace": NS_WITNESS,
        }
    )

    RenewablesEnergyInvestmentsValue = "Renewables energy investments [100G$]"
    RenewablesEnergyInvestments = {
        "var_name": RenewablesEnergyInvestmentsValue,
        "namespace": NS_WITNESS,
        "type": "dataframe",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            InvestmentsValue: ("float", [0.0, 1e30], True),
        },
        "unit": "100G$",
    }

    EnergyInvestmentsWoRenewable = {
        "var_name": EnergyInvestmentsWoRenewableValue,
        "type": "dataframe",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            EnergyInvestmentsWoRenewableValue: ("float", [0.0, 1e30], True),
        },
        "unit": "100G$",
    }

    ShareNonEnergyInvestment = {
        "var_name": ShareNonEnergyInvestmentsValue,
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
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
            Years: ("int", [1900, 2100], False),
            ExoGForcing: ("float", None, False),
            Forcing: ("float", None, False),
            TempAtmo: ("float", None, False),
            TempOcean: ("float", None, False),
        },
    }

    UtilityDiscountRate = "u_discount_rate"
    PeriodUtilityPerCapita = "period_utility_pc"
    DiscountedUtility = "discounted_utility"
    Welfare = "welfare"
    EnergyPriceRatio = "energy_price_ratio"
    PerCapitaConsumptionUtility = "Per capita consumption utility"
    UtilityDf = {
        "var_name": UtilityDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            UtilityDiscountRate: ("float", None, False),
            PeriodUtilityPerCapita: ("float", None, False),
            DiscountedUtility: ("float", None, False),
            EnergyPriceRatio: ("float", None, False),
            PerCapitaConsumptionUtility: ("float", None, False),
        },
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
            Years: ("int", [1900, 2100], False),
            GrossOutput: ("float", None, False),
            OutputNetOfDamage: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            Capital: ("float", None, False),
            UsableCapital: ("float", None, False),
        },
    }

    EnergyCapitalDfValue = "energy_capital"
    EnergyCapitalDf = {
        "var_name": EnergyCapitalDfValue,
        "type": "dataframe",
        "unit": "G$",
        "description": "Capital of energy in G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            Capital: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            Capital: ("float", None, False),
            UsableCapital: ("float", None, False),
            Emax: ("float", None, False),
            EnergyEfficiency: ("float", None, False),
        },
    }

    SectorizedEconomicsDf = {  # todo: miss per capita consumption !
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            GrossOutput: ("float", None, False),
            OutputNetOfDamage: ("float", None, False),
            Capital: ("float", None, False),
        },
    }

    SectorizedEconomicsDetailDf = {  # todo: miss per capita consumption !
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            GrossOutput: ("float", None, False),
            OutputNetOfDamage: ("float", None, False),
            Capital: ("float", None, False),
            UsableCapital: ("float", None, False),
            OutputGrowth: ("float", None, False),
            Damages: ("float", None, False),
            Consumption: ("float", None, False),
        },
    }

    ProductivityDfValue = "productivity_df"
    ProductivityDf = {
        "var_name": ProductivityDfValue,
        "type": "dataframe",
        "unit": "-",
        "description": "productivity levels through years, applied, with damage, and without wamage.",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            Productivity: ("float", None, False),
            ProductivityGrowthRate: ("float", None, False),
            ProductivityWithoutDamage: ("float", None, False),
            ProductivityWithDamage: ("float", None, False),
            OptimalEnergyProduction: ("float", None, False),
            UsedEnergy: ("float", None, False),
            UnusedEnergy: ("float", None, False),
            EnergyWasted: ("float", None, False)
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
            Years: ("int", [1900, 2100], False),
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
            Years: ("int", [1900, 2100], False),
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
            Years: ("int", [1900, 2100], False),
            ShareSectorEnergy: ("float", [0.0, 100.0], False),
        },
    }
    ResidentialEnergyProductionDfValue = "residential_energy_production_df"
    ResidentialEnergyProductionDf = {
        "var_name": RedistributionEnergyProductionDfValue,
        "type": "dataframe",
        "unit": "PWh",
        "visibility": "Shared",
        "namespace": NS_WITNESS,
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            TotalProductionValue: ("float", None, False),
        },
    }

    OtherEnergyCategory = "Other"
    ShareOtherEnergyDfValue = "share_other_energy_df"
    ShareOtherEnergyDf = {
        "type": "dataframe",
        "unit": "%",
        "description": "Amount of the total energy production attributed to other category",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
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
            Years: ("float", None, False),
            Population1570: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            InvestmentsValue: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            SectorDemandPerCapitaDfValue: ("float", None, False),
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
            Years: ("int", [1900, 2100], False),
            SectorGDPDemandDfValue: ("float", None, False),
        },
    }

    InvestmentShareGDPValue = "total_investment_share_of_gdp"
    InvestmentShareGDP = {
        "var_name": InvestmentShareGDPValue,
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "share_investment": ("float", None, True),
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
            "invest": ("float", None, True),
        },
        "dataframe_edition_locked": False,
    }

    ShareMaxInvestName = "share_max_invest"
    ShareMaxInvest = {
        "var_name": ShareMaxInvestName,
        "type": "float",
        "unit": "%",
        "default": 10. ,
        "description": "float to set maximum percentage of GDP to allow to investments in sectors and energy"
    }

    UtilisationRatioValue = "Utilisation Ratio [%]"

    MaxInvestConstraintName = "max_invest_constraint"
    MaxInvestConstraint = {
        "var_name": MaxInvestConstraintName,
        "type": "array",
        "unit": "[]",
        "description": "Max investment in sectors constraint using share_max_invest percentage"
    }

    MaxInvestConstraintRefName = "max_invest_constraint_ref"
    MaxInvestConstraintRef = {
        "var_name": MaxInvestConstraintRefName,
        "type": "float",
        "unit": "G$",
        "default": 100.,
        "user_level": 3,
        "description": "Max investment reference to normalize associated constraint"
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
