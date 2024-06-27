'''
Copyright 2024 Capgemini

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

import numpy as np
import plotly.graph_objects as go
import re
from energy_models.glossaryenergy import GlossaryEnergy as Glossary
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

from climateeconomics.core.core_land_use.land_use_v2 import LandUseV2
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.sos_wrapping.post_procs.iea_data_preparation.iea_data_preparation import (
    IEADataPreparation,
)


def update_variable_name(list_var_value, suffix):
    """
    Update variable name and add an additional extension to variable name and to variable name in var_value
    """
    dict_in = {}
    dict_out = {}
    for var_value in list_var_value:
        # get dict to update
        variable_dict = Glossary.get_dynamic_variable(var_value)
        # delete namespace and visibility if existing
        variable_dict.pop('namespace', None)
        variable_dict.pop('visibility', None)
        var_name = variable_dict['var_name']
        # copy dict before storing it as we do a modification later
        dict_in[var_name] = variable_dict.copy()
        var_name_updated = var_name + suffix
        variable_dict['var_name'] = var_name_updated
        dict_out[var_name_updated] = variable_dict
    return dict_in, dict_out


def create_production_variables(list_technologies):
    """
    Create production variables based on input list
    """
    dict_in = {}
    for techno_name in list_technologies:
        var_name = f'{techno_name}_{Glossary.TechnoProductionValue}'
        dict_in[var_name] = Glossary.get_dynamic_variable(Glossary.TechnoProductionDf)
        dict_in[var_name]['var_name'] = var_name
    return dict_in


class IEADataPreparationDiscipline(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'IEA Data preparation discipline',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    _maturity = 'Research'

    SUFFIX_VAR_INTERPOLATED = '_interpolated'  # var_out = var_in + SUFFIX_VAR_INTERPOLATED
    IEA_NAME = 'IEA'

    # all input variables are in output as well, only difference is that we add _interpolated to output variables
    years = np.arange(Glossary.YearStartDefault, Glossary.YearEndDefault + 1)
    # get variables from glossary to update either namespace or dataframe descriptor
    co2_emissions_dict_variable = Glossary.get_dynamic_variable(Glossary.CO2EmissionsGt)
    gdp_dict_variable = Glossary.get_dynamic_variable(Glossary.EconomicsDf)
    population_dict_variable = Glossary.get_dynamic_variable(Glossary.PopulationDf)
    co2_tax_dict_value = Glossary.get_dynamic_variable(Glossary.CO2Taxes)
    energy_production_dict_value = Glossary.get_dynamic_variable(Glossary.EnergyProductionDf)
    temperature_dict_value = Glossary.get_dynamic_variable(Glossary.TemperatureDf)
    land_use_surface_dict_value = {'var_name': LandUseV2.LAND_SURFACE_DETAIL_DF,
                                   'type': 'dataframe', 'unit': 'Gha',
                                   'dataframe_descriptor': {Glossary.Years: ('float', None, False),
                                                            'Crop (Gha)': ('float', None, False),
                                                            'Food Surface (Gha)': ('float', None, False),
                                                            'Total Agriculture Surface (Gha)': ('float', None, False),
                                                            'Total Forest Surface (Gha)': ('float', None, False),
                                                            }
                                   }
    # list of dictionaries to update
    l_dict_to_update = [co2_emissions_dict_variable, gdp_dict_variable, population_dict_variable, co2_tax_dict_value,
                        energy_production_dict_value, temperature_dict_value, land_use_surface_dict_value,
                        ]
    # compute the updated dictionaries to be used in both desc_in and desc_out
    desc_in_updated, desc_out_updated = update_variable_name(l_dict_to_update, SUFFIX_VAR_INTERPOLATED)

    # add techno production
    l_technos_to_add = [f'{Glossary.electricity}_{Glossary.Nuclear}',
                        f'{Glossary.electricity}_{Glossary.Hydropower}',
                        f'{Glossary.electricity}_{Glossary.Solar}',
                        f'{Glossary.electricity}_{Glossary.WindOnshoreAndOffshore}',
                        f'{Glossary.solid_fuel}_{Glossary.CoalExtraction}',
                        f'{Glossary.methane}_{Glossary.FossilGas}',
                        f'{Glossary.biogas}_{Glossary.AnaerobicDigestion}',
                        f'{Glossary.CropEnergy}',
                        f'{Glossary.ForestProduction}',
                        ]
    # get techno production metadata from glossary and modify them with the correct name
    dict_values_techno_production = create_production_variables(l_technos_to_add)
    dict_in_production, dict_out_production = update_variable_name(list(dict_values_techno_production.values()), SUFFIX_VAR_INTERPOLATED)
    # update created desc_in and desc_out with the new variables
    desc_in_updated.update(dict_in_production)
    desc_out_updated.update(dict_out_production)
    # add energy prices variable for electricity and natural gas
    for energy in [f'{Glossary.electricity}', f'{Glossary.methane}']:
        energy_prices_dict = Glossary.get_dynamic_variable(Glossary.EnergyPricesDf)
        # only energy price to compare is for electricity technologies. No need to call a function
        energy_prices_dict['var_name'] = f'{energy}_{Glossary.EnergyPricesValue}'
        desc_in_energy_prices, desc_out_energy_prices = update_variable_name([energy_prices_dict], SUFFIX_VAR_INTERPOLATED)
        # update desc_in and desc_out with energy prices variable
        desc_in_updated.update(desc_in_energy_prices)
        desc_out_updated.update(desc_out_energy_prices)
    # store list of input variables for later use
    variables_to_store = list(desc_in_updated.keys())
    # create list of units by hand as very generic in Glossary TODO use glossary later
    list_units = ['Gt', 'T$', 'millions of people', '$/tCO2Eq', 'TWh', '°C', 'Gha', 'TWh', 'TWh', 'TWh', 'TWh', 'TWh', 'TWh', 'TWh','TWh','TWh', '$/MWh', '$/MWh']
    variables_to_store_with_units = dict(zip(variables_to_store, list_units))

    DESC_IN = {
        Glossary.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        Glossary.YearEnd: Glossary.YearEndVar,
    }
    DESC_IN.update(desc_in_updated)

    DESC_OUT = desc_out_updated

    def init_execution(self):
        """
        Init execution of model
        """
        self.iea_data_preparation_model = IEADataPreparation(self.variables_to_store_with_units)

    def run(self):
        """
        Run discipline
        """
        # get input of discipline
        param_in = self.get_sosdisc_inputs()
        self.iea_data_preparation_model.configure_parameters(param_in)

        # compute output
        self.iea_data_preparation_model.compute(
        )

        # store data
        self.store_sos_outputs_values(self.iea_data_preparation_model.dict_df_out)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = []
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        data_out = self.get_sosdisc_outputs()
        data_in = self.get_sosdisc_inputs()

        # loop on all output keys
        for key, df_processed in data_out.items():
            # recompute data_in key (without _interpolated)
            original_key = key.replace(self.SUFFIX_VAR_INTERPOLATED, '')
            df_original = data_in[original_key]

            fig = go.Figure()
            # Plot original values with circles
            for col in df_original.columns:
                if col != Glossary.Years:
                    fig.add_trace(go.Scatter(
                        x=df_original[Glossary.Years].tolist(),
                        y=df_original[col].tolist(),
                        mode='markers',
                        name=f'{col} (IEA)',
                        marker=dict(symbol='circle', size=8)
                    ))
            # initialize unit variable
            unit = None
            # Plot interpolated and extrapolated values with dashed lines
            for col in df_processed.columns:
                if col != Glossary.Years:
                    fig.add_trace(go.Scatter(
                        x=df_processed[Glossary.Years].tolist(),
                        y=df_processed[col].tolist(),
                        mode='lines',
                        name=f'{col} (Interpolated/Extrapolated)',
                        line=dict(dash='dash')
                    ))
                    # by construction of the model, the units of all columns of a dataframe are the same
                    # we just need to get at least one column to get the unit
                    # unit is stored in all columns name (except years) between brackets [ ]
                    unit = self.preprocess_column_name_to_get_unit(col)
            # preprocess name of key to obtain unit of variable
            yaxis_title = unit
            if yaxis_title is None :
                yaxis_title = "Unit undefined"

            fig.update_layout(
                title=f'Interpolated and Extrapolated Data for {original_key}',
                xaxis_title=Glossary.Years,
                yaxis_title=yaxis_title,
            )
            new_chart = InstantiatedPlotlyNativeChart(fig, f'Interpolated and Extrapolated Data for {original_key}')
            instanciated_charts.append(new_chart)
        return instanciated_charts

    @staticmethod
    def preprocess_column_name_to_get_unit(col):
        """
        Get unit from column name
        All columns in the model are named as followed col [unit]
        The method extracts the unit
        col : string column name to extract
        """
        match = re.search(r'\[(.*?)\]', col)
        # return found unit in column name
        return f"[{match.group(1)}]" if match else None