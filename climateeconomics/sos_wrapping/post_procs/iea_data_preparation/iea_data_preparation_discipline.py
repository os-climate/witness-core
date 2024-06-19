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
from copy import deepcopy

import numpy as np

from climateeconomics.sos_wrapping.post_procs.iea_data_preparation.iea_data_preparation import IEADataPreparation
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from energy_models.glossaryenergy import GlossaryEnergy as Glossary
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)


def update_variable_name(list_var_value):
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
        var_name_updated = var_name + '_interpolated'
        variable_dict['var_name'] = var_name_updated
        dict_out[var_name_updated] = variable_dict
    return dict_in, dict_out


def create_production_variables(list_technologies):
    """
    Create production variables based on input list
    """
    dict_in = {}
    for techno_name in list_technologies:
        var_name = f'{techno_name}_techno_production'
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

    # all input variables are in output as well, only difference is that we add _interpolated to output variables
    years = np.arange(Glossary.YearStartDefault, Glossary.YearEndDefault + 1)
    # get variables from glossary to update either namespace or dataframe descriptor
    co2_emissions_dict_variable = Glossary.get_dynamic_variable(Glossary.CO2EmissionsGt)
    gdp_dict_variable = Glossary.get_dynamic_variable(Glossary.EconomicsDf)
    population_dict_variable = Glossary.get_dynamic_variable(Glossary.PopulationDf)
    co2_tax_dict_value = Glossary.get_dynamic_variable(Glossary.CO2Taxes)
    energy_production_dict_value = Glossary.get_dynamic_variable(Glossary.EnergyProductionDf)
    temperature_dict_value = Glossary.get_dynamic_variable(Glossary.TemperatureDf)
    # list of dictionaries to update
    l_dict_to_update = [co2_emissions_dict_variable, gdp_dict_variable, population_dict_variable, co2_tax_dict_value,
                        energy_production_dict_value, temperature_dict_value]
    # compute the updated dictionaries to be used in both desc_in and desc_out
    desc_in_updated, desc_out_updated = update_variable_name(l_dict_to_update)

    # add techno production
    l_technos_to_add = [f'{Glossary.electricity}_{Glossary.Nuclear}', f'{Glossary.electricity}_{Glossary.Hydropower}',
                        f'{Glossary.electricity}_{Glossary.Solar}',
                        f'{Glossary.electricity}_{Glossary.WindOnshoreAndOffshore}',
                        f'{Glossary.solid_fuel}_{Glossary.CoalExtraction}', f'{Glossary.methane}_{Glossary.FossilGas}',
                        f'{Glossary.biogas}_{Glossary.AnaerobicDigestion}', f'{Glossary.CropEnergy}',
                        f'{Glossary.ForestProduction}'
                        ]
    # dict_values

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
        self.iea_data_preparation_model = IEADataPreparation()

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

        return instanciated_charts
