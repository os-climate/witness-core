class GlossaryCore:
    """Glossary gathering variables used in witness core"""

    # todo : Continue filling gathering values here ..
    Years = "years"

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
        "var_name": "economics_df",
        "type": "dataframe",
        "visibility": "Shared",
        "namespace": "ns_witness",
        "unit": "-",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "gross_output": ("float", None, False),
            "output_net_of_d": ("float", None, False),
            "net_output": ("float", None, False),
            "population": ("float", None, False),
            "productivity": ("float", None, False),
            "productivity_gr": ("float", None, False),
            "energy_productivity_gr": ("float", None, False),
            "energy_productivity": ("float", None, False),
            "consumption": ("float", None, False),
            "pc_consumption": ("float", None, False),
            "capital": ("float", None, False),
            "investment": ("float", None, False),
            "interest_rate": ("float", None, False),
            "energy_investment": ("float", None, False),
            "non_energy_investment": ("float", None, False),
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

    ShareEnergyInvestment = {
        "var_name": "share_energy_investment",
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "energy": ("float", [0.0, 100.0], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": "ns_witness",
    }

    ShareNonEnergyInvestment = {
        "var_name": "share_non_energy_investment",
        "type": "dataframe",
        "unit": "%",
        "dataframe_descriptor": {
            Years: ("float", None, False),
            "non_energy": ("float", [0.0, 100.0], True),
        },
        "dataframe_edition_locked": False,
        "visibility": "Shared",
        "namespace": "ns_witness",
    }
