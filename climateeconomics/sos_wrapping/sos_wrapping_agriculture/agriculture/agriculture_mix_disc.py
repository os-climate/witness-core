'''
Copyright 2022 Airbus SAS

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

from energy_models.core.stream_type.energy_disc import EnergyDiscipline
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart


class AgricultureMixDiscipline(EnergyDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'Agriculture Mix Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-wheat',
        'version': '',
    }

    DESC_IN = {'technologies_list': {'type': 'string_list',
                                     'possible_values': ['Crop', 'Forest'],
                                     'visibility': EnergyDiscipline.SHARED_VISIBILITY,
                                     'namespace': 'ns_agriculture',
                                     'structuring': True},
               'data_fuel_dict': {'type': 'dict', 'visibility': EnergyDiscipline.SHARED_VISIBILITY,
                                  'namespace': 'ns_agriculture', 'default': BiomassDry.data_energy_dict},
               }
    DESC_IN.update(EnergyDiscipline.DESC_IN)
    name = 'AgricultureMix'
    energy_name = BiomassDry.name

    # -- add specific techno outputs to this
    DESC_OUT = EnergyDiscipline.DESC_OUT

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.energy_model = BiomassDry(self.energy_name)
        self.energy_model.configure_parameters(inputs_dict)

    def get_chart_co2_emissions(self):
        new_charts = []
        # chart_name = f'Comparison of CO2 emissions due to production and use<br>of {self.energy_name} technologies'
        # new_chart = TwoAxesInstanciatedChart(
        #     'years', 'CO2 emissions (Mt)', chart_name=chart_name, stacked_bar=True)

        # technology_list = self.get_sosdisc_inputs('technologies_list')

        # co2_per_use = self.get_sosdisc_outputs(
        #     'CO2_per_use')

        # energy_production = self.get_sosdisc_outputs('energy_production')
        # scaling_factor_energy_production = self.get_sosdisc_inputs(
        #     'scaling_factor_energy_production')
        # for technology in technology_list:
        #     techno_emissions = self.get_sosdisc_inputs(
        #         f'{technology}.CO2_emissions')
        #     year_list = techno_emissions['years'].values.tolist()
        #     emission_list = techno_emissions[technology].values * \
        #         energy_production[self.energy_name].values * \
        #         scaling_factor_energy_production
        #     serie = InstanciatedSeries(
        #         year_list, emission_list.tolist(), technology, 'bar')
        #     new_chart.series.append(serie)

        # co2_per_use = co2_per_use['CO2_per_use'].values * \
        #     energy_production[self.energy_name].values * \
        #     scaling_factor_energy_production
        # serie = InstanciatedSeries(
        #     year_list, co2_per_use.tolist(), 'CO2 from use of brut production', 'bar')
        # new_chart.series.append(serie)
        # new_charts.append(new_chart)

        return new_charts