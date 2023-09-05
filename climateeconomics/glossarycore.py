class GlossaryCore:
    """Glossary gathering variables used in witness core"""

    # todo : Continue filling gathering values here ..
    # dataframes columns/var_names :
    Years = "years"
    InvestmentsValue = "investment"
    ShareNonEnergyInvestmentsValue = "share_non_energy_investment"
    EconomicsDfValue = "economics_df"
    EnergyInvestmentsValue = "energy_investment"
    EnergyInvestmentsWoTaxValue = "energy_investment_wo_tax"
    EnergyInvestmentsWoRenewableValue = 'energy_investment_wo_renewable'
    NonEnergyInvestmentsValue = "non_energy_investment"
    EnergyInvestmentsFromTaxValue = "energy_investment_from_tax"

    CO2EmissionsGt = {
        "var_name": "co2_emissions_Gt",
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_energy_mix",
        "unit": "Gt",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "Total CO2 emissions": ("float", None, False),
            "cumulative_total_energy_supply": ("float", None, False),
        },
    }

    CO2Taxes = {
        "var_name": "CO2_taxes",
        "type": "dataframe",
        "unit": "$/tCO2",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "dataframe_descriptor": {
            Years: ("int", [1900, 2100], False),
            "CO2_tax": ("float", None, True),
        },
        "dataframe_edition_locked": False,
    }

    DietMortalityParamDf = {'var_name': 'diet_mortality_param_df',
                            'type': 'dataframe',
                            'default': 'default_diet_mortality_param_df', 'user_level': 3,
                            'unit': '-',
                            'dataframe_descriptor': {'param': ('string', None, False),
                                                     'undernutrition': ('float', None, True),
                                                     'overnutrition': ('float', None, True),
                                                     }
                            }

    DamageDf = {
        "var_name": "damage_df",
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "unit": "G$",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "damages": ("float", None, False),
            "damage_frac_output": ("float", None, False),
            "base_carbon_price": ("float", None, False),
        },
    }

    Economics_df = {
        "var_name": EconomicsDfValue,
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "gross_output": ("float", None, False),
            "output_net_of_d": ("float", None, False),
            "pc_consumption": ("float", None, False),
        },
    }

    EconomicsDetail_df = {
        "var_name": 'economics_detail_df',
        'type': 'dataframe',
        'unit': '-',
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "gross_output": ("float", None, False),
            "output_net_of_d": ("float", None, False),
            "productivity": ("float", None, False),
            "productivity_gr": ("float", None, False),
            "consumption": ("float", None, False),
            "pc_consumption": ("float", None, False),
            InvestmentsValue: ("float", None, False),
            EnergyInvestmentsValue: ("float", None, False),
            EnergyInvestmentsWoTaxValue: ("float", None, False),
            NonEnergyInvestmentsValue: ("float", None, False),
            EnergyInvestmentsFromTaxValue: ("float", None, False),
            "output_growth": ("float", None, False),
        },
    }

    PopulationDF = {
        "var_name": "population_df",
        "type": "dataframe",
        "unit": "millions of people",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "population": ("float", None, False),
        },
    }

    EnergyInvestments = {
        "var_name": EnergyInvestmentsValue,
        "type": "dataframe",
        "unit": '100G$',
        "dataframe_descriptor": {
            Years: ("float", None, False),
            EnergyInvestmentsValue: ("float", [0.0, 1e30], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": "ns_witness",
    }

    EnergyInvestmentsWoTax = {
        "var_name": EnergyInvestmentsWoTaxValue,
        "type": "dataframe",
        "unit": 'G$',
        "dataframe_descriptor": {
            Years: ("float", None, False),
            EnergyInvestmentsWoTaxValue: ("float", [0.0, 1e30], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": "ns_witness",
    }
    EnergyInvestmentsWoRenewable = {
        "var_name": EnergyInvestmentsWoRenewableValue,
        'type': 'dataframe',
        "dataframe_descriptor": {
            Years: ("float", None, False),
            EnergyInvestmentsWoRenewableValue: ("float", [0.0, 1e30], True),
        },
        'unit': '100G$'
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
    TemperatureDf = {'var_name':'temperature_df',
                     'type': 'dataframe', 'visibility': 'Shared', 'namespace': 'ns_witness',
                     'unit': 'degree Celsius',
                      'dataframe_descriptor': {Years: ('float', None, False),
                                                'exog_forcing': ('float', None, False),
                                                'forcing': ('float', None, False),
                                                'temp_atmo': ('float', None, False),
                                                'temp_ocean': ('float', None, False),
                                               }
                       }
