'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/28-2023/11/03 Copyright 2023 Capgemini

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
import pandas as pd

from climateeconomics.core.core_witness.climateeco_discipline import ClimateEcoDiscipline
from climateeconomics.core.core_witness.damage_model import DamageModel
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart


class DamageDiscipline(ClimateEcoDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'Damage WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-exclamation-triangle fa-fw',
        'version': '',
    }

    years = np.arange(2020, 2101)
    CO2_tax = np.asarray([500.] * len(years))
    default_CO2_tax = pd.DataFrame(
        {GlossaryCore.Years: years, GlossaryCore.CO2Tax: CO2_tax}, index=years)

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: ClimateEcoDiscipline.YEAR_END_DESC_IN,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        'init_damag_int': {'type': 'float', 'default': 0.0, 'unit': '-', 'user_level': 3},
        'damag_int': {'type': 'float', 'default': 0.0, 'unit': '-', 'user_level': 3},
        'damag_quad': {'type': 'float', 'default': 0.0022, 'unit': '-', 'user_level': 3},
        'damag_expo': {'type': 'float', 'default': 2.0, 'unit': '-', 'user_level': 3},
        'tipping_point': {'type': 'bool', 'default': True},
        'tp_a1': {'type': 'float', 'visibility': ClimateEcoDiscipline.INTERNAL_VISIBILITY, 'default': 20.46, 'user_level': 3, 'unit': '-'},
        'tp_a2': {'type': 'float', 'visibility': ClimateEcoDiscipline.INTERNAL_VISIBILITY, 'default': 2, 'user_level': 3, 'unit': '-'},
        'tp_a3': {'type': 'float', 'visibility': ClimateEcoDiscipline.INTERNAL_VISIBILITY, 'default': 6.081, 'user_level': 3, 'unit': '-'},
        'tp_a4': {'type': 'float', 'visibility': ClimateEcoDiscipline.INTERNAL_VISIBILITY, 'default': 6.754, 'user_level': 3, 'unit': '-'},
        GlossaryCore.FractionDamageToProductivityValue: {'type': 'float', 'default': 0.30, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_witness', 'user_level': 2},
        GlossaryCore.DamageDfValue: GlossaryCore.DamageDf,
        GlossaryCore.TemperatureDfValue: GlossaryCore.TemperatureDf,
        'total_emissions_damage_ref': {'type': 'float', 'default': 18.0, 'unit': 'Gt', 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                       'namespace': 'ns_ref', 'user_level': 2},
        'damage_constraint_factor': {'type': 'array', 'unit': '-', 'user_level': 2},
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN
    }

    DESC_OUT = {
        'CO2_damage_price': {'type': 'dataframe', 'unit': '$/tCO2', 'visibility': 'Shared', 'namespace': 'ns_witness'},
        GlossaryCore.DamageFractionDfValue: GlossaryCore.DamageFractionDf,
    }

    _maturity = 'Research'

    def init_execution(self):
        in_dict = self.get_sosdisc_inputs()
        self.model = DamageModel(in_dict)

    def setup_sos_disciplines(self):
        """
        Check if flag 'compute_climate_impact_on_gdp' is on or not.
        If so, then the output GlossaryCore.DamageDf['var_name' is shared with other disciplines that requires it as input,
        else it is not, and therefore others discipline will demand to specify t input
        """

        dynamic_outputs = {}

        self.add_outputs(dynamic_outputs)

        self.update_default_with_years()

    def update_default_with_years(self):
        '''
        Update all default dataframes with years 
        '''
        if GlossaryCore.YearStart in self.get_data_in():
            year_start, year_end = self.get_sosdisc_inputs(
                [GlossaryCore.YearStart, GlossaryCore.YearEnd])
            years = np.arange(year_start, year_end + 1)
            damage_constraint_factor_default = np.concatenate(
                (np.linspace(1.0, 1.0, 20), np.asarray([1] * (len(years) - 20))))
            self.set_dynamic_default_values(
                {'damage_constraint_factor': damage_constraint_factor_default})

    def run(self):
        # get inputs
        in_dict = self.get_sosdisc_inputs()
        damage_df = in_dict.pop(GlossaryCore.DamageDfValue)
        temperature_df = in_dict.pop(GlossaryCore.TemperatureDfValue)

        # pyworld3 execution
        damage_fraction_df, co2_damage_price_df = self.model.compute(
            damage_df, temperature_df)

        # store output data
        out_dict = {GlossaryCore.DamageFractionDfValue: damage_fraction_df[GlossaryCore.DamageFractionDf['dataframe_descriptor'].keys()],
                    'CO2_damage_price': co2_damage_price_df}

        self.store_sos_outputs_values(out_dict)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradiant of coupling variable to compute: 
        damage_df
          - GlossaryCore.Damages:
                - temperature_df, GlossaryCore.TempAtmo
                - economics_df, GlossaryCore.GrossOutput
          -GlossaryCore.DamageFractionOutput
                - temperature_df, GlossaryCore.TempAtmo
        """
        ddamage_frac_output_temp_atmo = self.model.compute_gradient()
        d_co2_damage_price_d_damages = self.model.d_co2_damage_price_d_damages()
        # fill jacobians

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.DamageFractionDfValue, GlossaryCore.DamageFractionOutput),
            (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo),
            ddamage_frac_output_temp_atmo)
        self.set_partial_derivative_for_other_types(
            ('CO2_damage_price', 'CO2_damage_price'),
            (GlossaryCore.DamageDfValue, GlossaryCore.Damages),
            d_co2_damage_price_d_damages)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [GlossaryCore.Damages, 'CO2 damage price']  # , 'Abatement cost']
        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if GlossaryCore.Damages in chart_list:

            damage_fraction_df = deepcopy(self.get_sosdisc_outputs(GlossaryCore.DamageFractionDfValue))
            years = list(damage_fraction_df.index)

            chart_name = 'Lost GDP due to climate damages [%]'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '%', chart_name=chart_name, y_min_zero=True)

            new_series = InstanciatedSeries(
                years, list(damage_fraction_df[GlossaryCore.DamageFractionOutput]),
                'Climate damage on GDP', 'lines', True)

            new_chart.add_series(new_series)

            note = {'Note': 'Damages due to loss of productivity do not appear here'}
            new_chart.annotation_upper_left = note

            instanciated_charts.append(new_chart)

        if 'CO2 damage price' in chart_list:

            co2_damage_price_df = deepcopy(
                self.get_sosdisc_outputs('CO2_damage_price'))

            co2_damage_price = co2_damage_price_df['CO2_damage_price']

            years = list(co2_damage_price_df[GlossaryCore.Years].values.tolist())

            chart_name = 'CO2 damage price'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Price ($/tCO2)', chart_name=chart_name, y_min_zero=True)

            note = {'Note': 'Damages due to loss of productivity are not included'}
            new_chart.annotation_upper_left = note

            visible_line = True

            # add CO2 damage price serie
            new_series = InstanciatedSeries(
                years, co2_damage_price.values.tolist(), 'CO2 damage price', 'lines', visible_line)
            new_chart.add_series(new_series)

            # add chart
            instanciated_charts.append(new_chart)

        if True:
            import numpy as np

            tp_a1 = self.get_sosdisc_inputs("tp_a1")
            tp_a2 = self.get_sosdisc_inputs("tp_a2")
            tp_a3 = self.get_sosdisc_inputs("tp_a3")
            tp_a4 = self.get_sosdisc_inputs("tp_a4")

            def damage_function_tipping_point(temp_increase):
                return (temp_increase / tp_a1) ** tp_a2 + (temp_increase / tp_a3) ** tp_a4

            def damage_fraction(damage):
                return damage / (1 + damage) * 100

            temperature_increase = np.linspace(0, 8, 100)

            damage_frac = damage_fraction(damage_function_tipping_point(temperature_increase))

            chart_name = "Tipping point damage model (Weitzman, 2009)"
            new_chart = TwoAxesInstanciatedChart('Temperature increase (°C)',
                                                 'Impact on GDP (%)',
                                                 chart_name=chart_name)

            new_series = InstanciatedSeries(list(temperature_increase), list(damage_frac),
                'Climate damage on GDP', 'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if True:
            damag_int = self.get_sosdisc_inputs("damag_int")
            damag_quad = self.get_sosdisc_inputs("damag_quad")

            def damage_function_tipping_point(temp_increase):
                return damag_int * temp_increase + damag_quad * temp_increase ** 2

            def damage_fraction(damage):
                return damage / (1 + damage) * 100

            temperature_increase = np.linspace(0, 8, 100)

            damage_frac = damage_fraction(damage_function_tipping_point(temperature_increase))

            chart_name = "Standard DICE damage model (Nordhaus, 2017)"
            new_chart = TwoAxesInstanciatedChart('Temperature increase (°C)',
                                                 'Impact on GDP (%)',
                                                 chart_name=chart_name)

            new_series = InstanciatedSeries(list(temperature_increase), list(damage_frac),
                'Climate damage on GDP', 'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
