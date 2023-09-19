from copy import deepcopy


class GlossaryCore:
    """Glossary gathering variables used in witness core"""

    # TODO, FILLING METHOD : goal is to spread the use of the glossary in the code, but we have to do it #
    #  methodically. Suggested method : every time you add a string/variable in the glossary, replace this string
    #  everywhere in the code. Only then, add continue filling the Glossary.
    # dataframes columns/var_names :
    Years = "years"
    YearStart = "year_start"
    YearEnd = "year_end"
    TimeStep = "time_step"
    # todo in the futur: merge these 3 invest values
    InvestValue = "invest"
    InvestLevelValue = "invest_level"
    InvestmentsValue = "investment"

    ShareNonEnergyInvestmentsValue = "share_non_energy_investment"
    CO2EmissionsGtValue = "co2_emissions_Gt"
    CO2TaxesValue = "CO2_taxes"
    DamageDfValue = "damage_df"
    EconomicsDfValue = "economics_df"
    PopulationDfValue = "population_df"
    TemperatureDfValue = "temperature_df"
    UtilityDfValue = "utility_df"
    EnergyInvestmentsValue = "energy_investment"
    EnergyInvestmentsWoTaxValue = "energy_investment_wo_tax"
    EnergyInvestmentsWoRenewableValue = "energy_investment_wo_renewable"
    NonEnergyInvestmentsValue = "non_energy_investment"
    EnergyInvestmentsFromTaxValue = "energy_investment_from_tax"  # T$

    energy_list = "energy_list"
    techno_list = "technologies_list"
    ccs_list = "ccs_list"

    invest_mix = "invest_mix"

    CarbonCycleDfValue = "carboncycle_df"
    CarbonCycleDf = {
        "var_name": CarbonCycleDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "atmo_conc": ("float", None, False),
            "lower_ocean_conc": ("float", None, False),
            "shallow_ocean_conc": ("float", None, False),
            "ppm": ("float", None, False),
            "atmo_share_since1850": ("float", None, False),
            "atmo_share_sinceystart": ("float", None, False),
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
            Years: ("float", None, False),
            TotalCO2Emissions: ("float", None, False),
            "cumulative_total_energy_supply": ("float", None, False),
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
            Years: ("float", None, False),
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

    GrossOutput = "gross_output"  # trillion $
    OutputNetOfDamage = "output_net_of_d"  # trillion $
    Consumption = "consumption"
    PerCapitaConsumption = "pc_consumption"
    EconomicsDf = {
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            GrossOutput: ("float", None, False),
            OutputNetOfDamage: ("float", None, False),
            PerCapitaConsumption: ("float", None, False),
        },
    }

    EconomicsDetailDfValue = "economics_detail_df"
    Productivity = "productivity"
    ProductivityGrowthRate = "productivity_gr"
    EconomicsDetail_df = {
        "var_name": EconomicsDetailDfValue,
        "type": "dataframe",
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            GrossOutput: ("float", None, False),  # T$
            OutputNetOfDamage: ("float", None, False),  # T$
            Productivity: ("float", None, False),
            ProductivityGrowthRate: ("float", None, False),
            Consumption: ("float", None, False),  # T$
            PerCapitaConsumption: ("float", None, False),
            InvestmentsValue: ("float", None, False),  # T$
            EnergyInvestmentsValue: ("float", None, False),  # T$
            EnergyInvestmentsWoTaxValue: ("float", None, False),  # T$
            NonEnergyInvestmentsValue: ("float", None, False),  # T$
            EnergyInvestmentsFromTaxValue: ("float", None, False),  # T$
            "output_growth": ("float", None, False),
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
            Years: ("float", None, False),
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
            Years: ("float", None, False),
            EnergyPriceValue: ("float", None, True),
        },
    }

    EnergyProductionValue = "energy_production"
    TotalProductionValue = "Total production"
    EnergyProduction = {
        "var_name": EnergyProductionValue,
        "type": "dataframe",
        "visibility": "Shared",
        "unit": "PWh",
        "namespace": "ns_energy_mix",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            TotalProductionValue: ("float", None, False),
        },
    }

    EnergyInvestments = {
        "var_name": EnergyInvestmentsValue,
        "type": "dataframe",
        "unit": "100G$",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            EnergyInvestmentsValue: ("float", [0.0, 1e30], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": "ns_witness",
    }

    EnergyInvestmentsWoTax = (
        {  # output of IndependentInvestDiscipline & input of MacroeconomicsDiscipline
            "var_name": EnergyInvestmentsWoTaxValue,
            "type": "dataframe",
            "unit": "T$",
            "dataframe_descriptor": {
                Years: ("float", None, False),
                EnergyInvestmentsWoTaxValue: ("float", [0.0, 1e30], True),
            },
            "dataframe_edition_locked": False,
            "visibility": "Shared",
            "namespace": "ns_witness",
        }
    )
    EnergyInvestmentsWoRenewable = {
        "var_name": EnergyInvestmentsWoRenewableValue,
        "type": "dataframe",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            EnergyInvestmentsWoRenewableValue: ("float", [0.0, 1e30], True),
        },
        "unit": "100G$",
    }

    ShareNonEnergyInvestment = {
        "var_name": ShareNonEnergyInvestmentsValue,
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("float", None, False),
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
            Years: ("float", None, False),
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
    UtilityDf = {
        "var_name": UtilityDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            UtilityDiscountRate: ("float", None, False),
            PeriodUtilityPerCapita: ("float", None, False),
            DiscountedUtility: ("float", None, False),
            Welfare: ("float", None, False),
        },
        "unit": "-",
    }

    SectorizedProductionDfValue = "production_df"
    SectorizedProductionDf = {
        "var_name": SectorizedProductionDfValue,
        "type": "dataframe",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            GrossOutput: ("float", None, False),
        },
    }

    SectorizedCapitalDfValue = "capital_df"
    Capital = "capital"
    UsableCapital = "usable_capital"
    SectorizedCapitalDf = {
        "var_name": SectorizedCapitalDfValue,
        "type": "dataframe",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            Capital: ("float", None, False),
            UsableCapital: ("float", None, False),
        },
    }

    SectorizedDetailedCapitalDfValue = "detailed_capital_df"
    Emax = "e_max"
    EnergyEfficiency = "energy_efficiency"
    SectorizedDetailedCapitalDf = {
        "var_name": SectorizedDetailedCapitalDfValue,
        "type": "dataframe",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            Capital: ("float", None, False),
            UsableCapital: ("float", None, False),
            Emax: ("float", None, False),
            EnergyEfficiency: ("float", None, False),
        },
    }

    SectorizedProductivityDfValue = "productivity_df"
    SectorizedProductivityDf = {
        "var_name": SectorizedCapitalDfValue,
        "type": "dataframe",
        "unit": "T$",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            Productivity: ("float", None, False),
            ProductivityGrowthRate: ("float", None, False),
        },
    }

    @staticmethod
    def delete_namespace(variable: dict):
        """delete the namespace of variable"""
        out = deepcopy(variable)
        try:
            del out["namespace"]
        except KeyError:
            pass
        return out

    @staticmethod
    def set_namespace(variable: dict, namespace: str):
        """set the namespace for a variable"""
        out = deepcopy(variable)
        out["namespace"] = namespace
        return out
