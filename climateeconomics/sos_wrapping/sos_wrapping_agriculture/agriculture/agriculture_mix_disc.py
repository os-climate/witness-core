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
import pandas as pd
import numpy as np
from climateeconomics.core.core_forest.forest_v2 import Forest
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
        'icon': 'fas fa-tractor fa-fw',
        'version': '',
    }

    DESC_IN = {'technologies_list': {'type': 'string_list',
                                     'possible_values': ['Crop', 'Forest'],
                                     'visibility': EnergyDiscipline.SHARED_VISIBILITY,
                                     'namespace': 'ns_agriculture',
                                     'structuring': True},
               'data_fuel_dict': {'type': 'dict', 'unit': 'defined in dict', 'visibility': EnergyDiscipline.SHARED_VISIBILITY,
                                  'namespace': 'ns_agriculture', 'default': BiomassDry.data_energy_dict},
               }
    DESC_IN.update(EnergyDiscipline.DESC_IN)
    name = 'AgricultureMix'
    energy_name = BiomassDry.name

    # -- add specific techno outputs to this
    DESC_OUT = {'CO2_land_emissions': {'type': 'dataframe', 'unit': 'GtCO2',
                                       'visibility': EnergyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_witness'}}
    DESC_OUT.update(EnergyDiscipline.DESC_OUT)

    def init_execution(self):
        inputs_dict = self.get_sosdisc_inputs()
        self.energy_model = BiomassDry(self.energy_name)
        self.energy_model.configure_parameters(inputs_dict)

    def setup_sos_disciplines(self):
        EnergyDiscipline.setup_sos_disciplines(self)
        dynamic_inputs = {}
        if 'technologies_list' in self._data_in:
            techno_list = self.get_sosdisc_inputs('technologies_list')
            if techno_list is not None:
                for techno in techno_list:
                    dynamic_inputs[f'{techno}.CO2_land_emission_df'] = {
                        'type': 'dataframe', 'unit': 'GtCO2'}
        self.add_inputs(dynamic_inputs, clean_inputs=False)

    def run(self):
        EnergyDiscipline.run(self)
        # -- get CO2 emissions inputs
        CO2_emitted_crop_df = self.get_sosdisc_inputs(
            'Crop.CO2_land_emission_df')
        CO2_emitted_forest_df = self.get_sosdisc_inputs(
            'Forest.CO2_land_emission_df')
        CO2_emissions_land_use_df = pd.DataFrame()
        CO2_emissions_land_use_df['years'] = CO2_emitted_crop_df['years']
        CO2_emissions_land_use_df['Crop'] = CO2_emitted_crop_df['emitted_CO2_evol_cumulative']
        CO2_emissions_land_use_df['Forest'] = CO2_emitted_forest_df['emitted_CO2_evol_cumulative']
        # -- store in one output
        self.store_sos_outputs_values(
            {'CO2_land_emissions': CO2_emissions_land_use_df})

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """
        EnergyDiscipline.compute_sos_jacobian(self)
        inputs_dict = self.get_sosdisc_inputs()
        self.energy_model.compute(inputs_dict)
        np_years = self.energy_model.year_end - self.energy_model.year_start + 1
        techno_list = self.get_sosdisc_inputs('technologies_list')
        for techno in techno_list:
            self.set_partial_derivative_for_other_types(
                ('CO2_land_emissions', f'{techno}'), (f'{techno}.CO2_land_emission_df', 'emitted_CO2_evol_cumulative'), np.identity(np_years))

    def get_chart_co2_emissions(self):
        new_charts = []
        chart_name = f'Comparison of CO2 emissions of agriculture lands'
        new_chart = TwoAxesInstanciatedChart(
            'years', 'CO2 emissions (GtCO2)', chart_name=chart_name, stacked_bar=True)

        technology_list = self.get_sosdisc_inputs('technologies_list')

        CO2_emissions_land_use_df = self.get_sosdisc_outputs(
            'CO2_land_emissions')
        year_start = self.get_sosdisc_inputs('year_start')
        year_end = self.get_sosdisc_inputs('year_end')
        year_list = np.arange(year_start, year_end + 1).tolist()
        for column in CO2_emissions_land_use_df.columns:
            if column != 'years':
                techno_emissions = CO2_emissions_land_use_df[column]
                serie = InstanciatedSeries(
                    year_list, techno_emissions.tolist(), column, 'bar')
                new_chart.series.append(serie)

        new_charts.append(new_chart)

        return new_charts
