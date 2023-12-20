'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/02-2023/11/03 Copyright 2023 Capgemini

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
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class ClimateEcoDiscipline(SoSWrapp):
    """
    Climate Economics Discipline
    """

    assumptions_dict_default = {'compute_gdp': True,
                                'compute_climate_impact_on_gdp': True,
                                'activate_climate_effect_population': True,
                                'invest_co2_tax_in_renewables': True,
                                }

    YEAR_START_DESC_IN = {'type': 'int', 'default': 2020,
                          'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_public'}
    YEAR_END_DESC_IN = {'type': 'int', 'default': 2100,
                        'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_public'}
    TIMESTEP_DESC_IN = {'type': 'int', 'default': 1, 'unit': 'year per period',
                        'visibility': 'Shared', 'namespace': 'ns_public', 'user_level': 2}
    ALPHA_DESC_IN = {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS,
                     'user_level': 1, 'unit': '-'}
    GWP_100_default = {'CO2': 1.0,
                       'CH4': 28.,
                       'N2O': 265.}

    GWP_20_default = {'CO2': 1.0,
                      'CH4': 85.,
                      'N2O': 265.}
    ASSUMPTIONS_DESC_IN = {
        'var_name': 'assumptions_dict', 'type': 'dict', 'default': assumptions_dict_default , 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'structuring': True, 'unit': '-'}

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Climate Economics Model',
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

    def get_greataxisrange(self, serie):
        """
        Get the lower and upper bound of axis for graphs 
        min_value: lower bound
        max_value: upper bound
        """
        min_value = serie.values.min()
        max_value = serie.values.max()
        min_range = self.get_value_axis(min_value, 'min')
        max_range = self.get_value_axis(max_value, 'max')

        return min_range, max_range

    def get_value_axis(self, value, min_or_max):
        """
        if min: if positive returns 0, if negative returns 1.1*value
        if max: if positive returns is 1.1*value, if negative returns 0
        """
        if min_or_max == 'min':
            if value >= 0:
                value_out = 0
            else:
                value_out = value * 1.1

        elif min_or_max == "max":
            if value >= 0:
                value_out = value * 1.1
            else:
                value_out = 0

        return value_out
