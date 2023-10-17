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

    SectorGdpPart = "Part of the GDP per sector [G$]"
    ChartSectorGDPPercentage = "Part of the GDP per sector [%]"

    ConstraintLowerBoundUsableCapital = "Lower bound usable capital constraint"
    EnergyWasted = "energy wasted [TWh]"
    EnergyWastedObjective = "energy_wasted_constraint"
    ShareNonEnergyInvestmentsValue = "share_non_energy_investment"
    CO2EmissionsGtValue = "co2_emissions_Gt"
    CO2TaxesValue = "CO2_taxes"
    DamageDfValue = "damage_df"
    EconomicsDfValue = "economics_df"
    SectorGdpDfValue = "sector_gdp_df"
    PopulationDfValue = "population_df"
    TemperatureDfValue = "temperature_df"
    UtilityDfValue = "utility_df"
    EnergyInvestmentsValue = "energy_investment"
    EnergyInvestmentsWoTaxValue = "energy_investment_wo_tax"
    EnergyInvestmentsWoRenewableValue = "energy_investment_wo_renewable"
    NonEnergyInvestmentsValue = "non_energy_investment"
    EnergyInvestmentsFromTaxValue = "energy_investment_from_tax"  # T$
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
        "namespace": "ns_witness",
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
        "namespace": "ns_witness",
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
    CO2EmissionsDetailDfValue = "CO2_emissions_detail_df"
    CO2EmissionsDfValue = "CO2_emissions_df"
    CO2EmissionsDetailDf = {
        "var_name": CO2EmissionsDetailDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
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
        "namespace": "ns_witness",
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
        "namespace": "ns_energy_mix",
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
        "namespace": "ns_witness",
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

    Damages = "damages"
    DamageFractionOutput = "damage_frac_output"
    BaseCarbonPrice = "base_carbon_price"
    DamageDf = {
        "var_name": DamageDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            Damages: ("float", None, False),
            DamageFractionOutput: ("float", None, False),
            BaseCarbonPrice: ("float", None, False),
        },
    }

    InitialGrossOutput = {
        "var_name": "init_gross_output",
        "type": "float",
        "unit": "T$",
        "visibility": "Shared",
        "default": 130.187,
        "namespace": "ns_witness",
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
        "namespace": "ns_witness",
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
        },
    }
    EconomicsDf = {
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
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
            GrossOutput: ("float", None, False),  # T$
            OutputNetOfDamage: ("float", None, False),  # T$
            Damages: ("float", None, False),  # T$
            Productivity: ("float", None, False),
            ProductivityGrowthRate: ("float", None, False),
            Consumption: ("float", None, False),  # T$
            PerCapitaConsumption: ("float", None, False),
            InvestmentsValue: ("float", None, False),  # T$
            EnergyInvestmentsValue: ("float", None, False),  # T$
            EnergyInvestmentsWoTaxValue: ("float", None, False),  # T$
            NonEnergyInvestmentsValue: ("float", None, False),  # T$
            EnergyInvestmentsFromTaxValue: ("float", None, False),  # T$
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
        "namespace": "ns_witness",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            PopulationValue: ("float", None, False),
        },
    }

    EnergyMeanPriceValue = "energy_mean_price"

    EnergyPriceValue = "energy_price"
    EnergyMeanPrice = {
        "var_name": EnergyMeanPriceValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_energy_mix",
        "unit": "$/MWh",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            EnergyPriceValue: ("float", None, True),
        },
    }

    EnergyProductionValue = "energy_production"
    TotalProductionValue = "Total production"
    EnergyProductionDf = {
        "var_name": EnergyProductionValue,
        "type": "dataframe",
        "visibility": "Shared",
        "unit": "PWh",
        "namespace": "ns_energy_mix",
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
        "namespace": "ns_sectors",
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
        "namespace": "ns_witness",
    }

    EnergyInvestmentsMinimizationObjective = "Energy invest minimization objective"
    EnergyInvestmentsWoTax = (
        {  # output of IndependentInvestDiscipline & input of MacroeconomicsDiscipline
            "var_name": EnergyInvestmentsWoTaxValue,
            "type": "dataframe",
            "unit": "T$",
            "dataframe_descriptor": {
                Years: ("int", [1900, 2100], False),
                EnergyInvestmentsWoTaxValue: ("float", [0.0, 1e30], True),
            },
            "dataframe_edition_locked": False,
            "visibility": "Shared",
            "namespace": "ns_witness",
        }
    )

    RenewablesEnergyInvestmentsValue = "Renewables energy investments [100G$]"
    RenewablesEnergyInvestments = {
        "var_name": RenewablesEnergyInvestmentsValue,
        "namespace": "ns_witness",
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
        "namespace": "ns_witness",
    }

    ExoGForcing = "exog_forcing"
    Forcing = "forcing"
    TempAtmo = "temp_atmo"
    TempOcean = "temp_ocean"
    TemperatureDf = {
        "var_name": TemperatureDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
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
        "namespace": "ns_witness",
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
        "namespace": "ns_sectors",
        "visibility": "Shared",
        "type": "dataframe",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            GrossOutput: ("float", None, False),
            OutputNetOfDamage: ("float", None, False),
        },
    }

    CapitalDfValue = "capital_df"
    Capital = "capital"
    UsableCapital = "usable_capital"
    UsableCapitalUnbounded = "Unbounded usable capital [T$]"
    NonEnergyCapital = "non_energy_capital"
    CapitalDf = {
        "var_name": CapitalDfValue,
        "namespace": "ns_witness",
        "visibility": "Shared",
        "type": "dataframe",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            Capital: ("float", None, False),
            UsableCapital: ("float", None, False),
        },
    }

    DetailedCapitalDfValue = "detailed_capital_df"
    Emax = "e_max"
    EnergyEfficiency = "energy_efficiency"
    DetailedCapitalDf = {
        "var_name": DetailedCapitalDfValue,
        "visibility": "Shared",
        "namespace": "ns_witness",
        "type": "dataframe",
        "unit": "T$",
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
        "namespace": "ns_witness",
        "unit": "T$",
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
        "var_name": CapitalDfValue,
        "type": "dataframe",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            Productivity: ("float", None, False),
            ProductivityGrowthRate: ("float", None, False),
        },
    }

    RedistributionInvestmentsDfValue = "redistribution_investments_df"
    RedistributionInvestmentsDf = {
        "var_name": RedistributionInvestmentsDfValue,
        "type": "dataframe",
        "unit": "T$",
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
        "namespace": "ns_sectors",
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
        "namespace": "ns_sectors",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            ShareSectorEnergy: ("float", [0.0, 100.0], False),
        },
    }

    ShareSectorEnergyDfValue = "share_sector_energy_df"
    ShareSectorEnergy = "Share of total energy production [%]"
    ShareSectorEnergyDf = {
        "type": "dataframe",
        "unit": "%",
        "description": "Amount of the total energy production attributed to the specific sector",
        "visibility": "Shared",
        "namespace": "ns_sectors",
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
        "namespace": "ns_witness",
    }

    WorkforceDfValue = "workforce_df"
    EmploymentRate = "employment_rate"
    Workforce = "workforce"
    WorkforceDf = {
        "var_name": WorkforceDfValue,
        "type": "dataframe",
        "unit": "millions of people",
        "visibility": "Shared",
        "namespace": "ns_witness",
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
        "namespace": "ns_witness",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            Population1570: ("float", None, False),
        },
    }

    InvestmentDfValue = "investment_df"
    InvestmentDf = {
        "var_name": InvestmentDfValue,
        "type": "dataframe",
        "unit": "T$",
        "visibility": "Shared",
        "namespace": "ns_sectors",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            InvestmentsValue: ("float", None, False),
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
        "namespace": "ns_witness",
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
