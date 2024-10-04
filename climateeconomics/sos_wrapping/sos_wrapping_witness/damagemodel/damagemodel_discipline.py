'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/28-2024/03/05 Copyright 2023 Capgemini

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
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.damage_model import DamageModel
from climateeconomics.glossarycore import GlossaryCore


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

    # fixme : dict of variables used to force update of namespace used only by post-proc modules OR namespace that only belongs to dynamic variables
    cheat_variables_dict = \
        {'cheat_var_to_update_ns_dashboard_in_ms_mdo': {'type': 'float','namespace':'ns_dashboard', 'visibility':'Shared', 'default': 0.0, 'unit': '-', 'user_level': 3},
        'cheat_var_to_update_ns_regions_in_ms_mdo': {'type': 'float', 'namespace': GlossaryCore.NS_REGIONALIZED_POST_PROC,
                                                       'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                                                       'user_level': 3},
        'cheat_var_to_update_ns_sectorspp_in_ms_mdo1': {'type': 'float',
                                                     'namespace': GlossaryCore.NS_SECTORS_POST_PROC_EMISSIONS,
                                                     'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                                                     'user_level': 3},
         'cheat_var_to_update_ns_sectorspp_in_ms_mdo2': {'type': 'float',
                                                         'namespace': GlossaryCore.NS_SECTORS_POST_PROC_GDP,
                                                         'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                                                         'user_level': 3},
        'cheat_var11': {'type': 'float', 'namespace': f'ns_{GlossaryCore.SectorIndustry.lower()}_emissions',
                                                        'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                                                        'user_level': 3},
        'cheat_varHOUSE': {'type': 'float', 'namespace': f'ns_{GlossaryCore.Households.lower()}_emissions',
                         'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                         'user_level': 3},
        'cheat_var12': {'type': 'float', 'namespace': f'ns_{GlossaryCore.SectorServices.lower()}_emissions',
                        'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                        'user_level': 3},
        #'cheat_var13': {'type': 'float', 'namespace': f'ns_{GlossaryCore.SectorAgriculture.lower()}_emissions',
        #                'visibility': 'Shared', 'default': 0.0, 'unit': '-',
        #                'user_level': 3},
        'cheat_var21': {'type': 'float', 'namespace': f'ns_{GlossaryCore.SectorIndustry.lower()}_gdp',
                        'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                        'user_level': 3},
        'cheat_var22': {'type': 'float', 'namespace': f'ns_{GlossaryCore.SectorServices.lower()}_gdp',
                        'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                        'user_level': 3},
        'cheat_var23': {'type': 'float', 'namespace': f'ns_{GlossaryCore.SectorAgriculture.lower()}_gdp',
                        'visibility': 'Shared', 'default': 0.0, 'unit': '-',
                        'user_level': 3}}
    DESC_IN = {
        **cheat_variables_dict,
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        'damag_int': {'type': 'float', 'default': 0.0, 'unit': '-', 'user_level': 3},
        'damag_quad': {'type': 'float', 'default': 0.0022, 'unit': '-', 'user_level': 3},
        'damag_expo': {'type': 'float', 'default': 2.0, 'unit': '-', 'user_level': 3},
        'tipping_point': {'type': 'bool', 'default': True},
        'tp_a1': {'type': 'float',  'default': 20.46, 'user_level': 3, 'unit': '-'},
        'tp_a2': {'type': 'float',  'default': 2, 'user_level': 3, 'unit': '-'},
        'tp_a3': {'type': 'float',  'default': 3.5, 'user_level': 3, 'unit': '-'},
        'tp_a4': {'type': 'float', 'default': 6.754, 'user_level': 3, 'unit': '-'},
        'total_emissions_damage_ref': {'type': 'float', 'default': 60.0, 'unit': 'Gt', 'user_level': 2},
        'co2_damage_price_dev_formula': {'type': 'bool', 'default': False, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        GlossaryCore.FractionDamageToProductivityValue: {'type': 'float', 'default': 0.30, 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'user_level': 2},
        GlossaryCore.DamageDfValue: GlossaryCore.DamageDf,
        GlossaryCore.TemperatureDfValue: GlossaryCore.TemperatureDf,
        GlossaryCore.ExtraCO2EqSincePreIndustrialValue: GlossaryCore.ExtraCO2EqSincePreIndustrialDf,
        'damage_constraint_factor': {'type': 'array', 'unit': '-', 'user_level': 2},
        'assumptions_dict': ClimateEcoDiscipline.ASSUMPTIONS_DESC_IN,
        GlossaryCore.CO2DamagePriceInitValue: GlossaryCore.CO2DamagePriceInitVar,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,


    }

    DESC_OUT = {
        GlossaryCore.CO2DamagePrice: GlossaryCore.CO2DamagePriceDf,
        GlossaryCore.DamageFractionDfValue: GlossaryCore.DamageFractionDf,
        GlossaryCore.ExtraCO2tDamagePrice: GlossaryCore.ExtraCO2tDamagePriceDf
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
            if year_start is not None and year_end is not None:
                years = np.arange(year_start, year_end + 1)
                damage_constraint_factor_default = np.concatenate(
                    (np.linspace(1.0, 1.0, 20), np.asarray([1] * (len(years) - 20))))
                self.set_dynamic_default_values(
                    {'damage_constraint_factor': damage_constraint_factor_default})

    def run(self):
        # get inputs
        in_dict = self.get_sosdisc_inputs()
        # todo: for sensitivity, generalise ?
        self.model.tp_a3 = in_dict['tp_a3']
        
        damage_df = in_dict.pop(GlossaryCore.DamageDfValue)
        temperature_df = in_dict.pop(GlossaryCore.TemperatureDfValue)
        extra_gigatons_co2_eq_df = in_dict.pop(GlossaryCore.ExtraCO2EqSincePreIndustrialValue)
        co2_damage_price_dev_formula = in_dict.pop("co2_damage_price_dev_formula")

        # pyworld3 execution
        damage_fraction_df, co2_damage_price_df, extra_co2_damage_price = self.model.compute(
            damage_df, temperature_df, extra_gigatons_co2_eq_df, co2_damage_price_dev_formula)

        # store output data
        out_dict = {GlossaryCore.DamageFractionDfValue: damage_fraction_df[GlossaryCore.DamageFractionDf['dataframe_descriptor'].keys()],
                    GlossaryCore.CO2DamagePrice: co2_damage_price_df,
                    GlossaryCore.ExtraCO2tDamagePrice: extra_co2_damage_price}

        
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
        d_extra_co2_t_damage_price_d_damages = self.model.d_extra_co2_t_damage_price_d_damages()
        d_extra_co2_t_damage_price_d_extra_co2_ton = self.model.d_extra_co2_t_damage_price_d_extra_co2_ton()

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.DamageFractionDfValue, GlossaryCore.DamageFractionOutput),
            (GlossaryCore.TemperatureDfValue, GlossaryCore.TempAtmo),
            ddamage_frac_output_temp_atmo)
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ExtraCO2tDamagePrice, GlossaryCore.ExtraCO2tDamagePrice),
            (GlossaryCore.ExtraCO2EqSincePreIndustrialValue, GlossaryCore.ExtraCO2EqSincePreIndustrialValue),
            self.model.d_extra_co2_t_damage_price_d_extra_co2_ton())
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ExtraCO2tDamagePrice, GlossaryCore.ExtraCO2tDamagePrice),
            (GlossaryCore.DamageDfValue, GlossaryCore.EstimatedDamages),
            self.model.d_extra_co2_t_damage_price_d_damages())

        co2_damage_price_dev_formula = self.get_sosdisc_inputs("co2_damage_price_dev_formula")
        if co2_damage_price_dev_formula:
            d_co2_damage_price_d_damages = self.model.d_co2_damage_price_dev_d_user_input(d_extra_co2_t_damage_price_d_damages)
            d_co2_damage_price_d_extra_co2_ton = self.model.d_co2_damage_price_dev_d_user_input(d_extra_co2_t_damage_price_d_extra_co2_ton)

            self.set_partial_derivative_for_other_types(
                (GlossaryCore.CO2DamagePrice, GlossaryCore.CO2DamagePrice),
                (GlossaryCore.DamageDfValue, GlossaryCore.EstimatedDamages),
                d_co2_damage_price_d_damages)
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.CO2DamagePrice, GlossaryCore.CO2DamagePrice),
                (GlossaryCore.ExtraCO2EqSincePreIndustrialValue, GlossaryCore.ExtraCO2EqSincePreIndustrialValue),
                d_co2_damage_price_d_extra_co2_ton)
        else:
            self.set_partial_derivative_for_other_types(
                (GlossaryCore.CO2DamagePrice, GlossaryCore.CO2DamagePrice),
                (GlossaryCore.DamageDfValue, GlossaryCore.EstimatedDamages),
                self.model.d_co2_damage_price_d_damages())

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = [GlossaryCore.Damages,
                      'CO2 damage price']
        co2_damage_price_dev_formula = self.get_sosdisc_inputs("co2_damage_price_dev_formula")
        if co2_damage_price_dev_formula:
            chart_list.append(GlossaryCore.ExtraCO2tDamagePrice)

        # First filter to deal with the view : program or actor
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        instanciated_charts = []
        chart_list = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        if GlossaryCore.Damages in chart_list:

            damage_fraction_df = deepcopy(self.get_sosdisc_outputs(GlossaryCore.DamageFractionDfValue))
            years = list(damage_fraction_df[GlossaryCore.Years].values)
            compute_climate_impact_on_gdp = self.get_sosdisc_inputs('assumptions_dict')['compute_climate_impact_on_gdp']
            chart_name = 'Lost GDP due to climate damages [%]' + ' (not applied)' * (not compute_climate_impact_on_gdp)
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, '%', chart_name=chart_name, y_min_zero=True)

            new_series = InstanciatedSeries(
                years, list(damage_fraction_df[GlossaryCore.DamageFractionOutput] * 100),
                'Climate damage on GDP', 'lines', True)

            new_chart.add_series(new_series)

            note = {'Note': 'This does not include damage due to loss of productivity'}
            new_chart.annotation_upper_left = note

            instanciated_charts.append(new_chart)

        if 'CO2 damage price' in chart_list:

            co2_damage_price_df = deepcopy(
                self.get_sosdisc_outputs(GlossaryCore.CO2DamagePrice))

            co2_damage_price = co2_damage_price_df[GlossaryCore.CO2DamagePrice]

            years = list(co2_damage_price_df[GlossaryCore.Years].values.tolist())

            chart_name = 'CO2eq damage price'

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, f'Price ({GlossaryCore.CO2DamagePriceDf["unit"]})', chart_name=chart_name, y_min_zero=True)

            visible_line = True

            # add CO2 damage price serie
            new_series = InstanciatedSeries(
                years, np.round(co2_damage_price.values, 2).tolist(), 'CO2 damage price', 'lines', visible_line)
            new_chart.add_series(new_series)

            # add chart
            instanciated_charts.append(new_chart)

        if GlossaryCore.ExtraCO2tDamagePrice in chart_list:
            extra_co2_damage_price_df = deepcopy(
                self.get_sosdisc_outputs(GlossaryCore.ExtraCO2tDamagePrice))

            extra_co2_damage_price = extra_co2_damage_price_df[GlossaryCore.ExtraCO2tDamagePrice].values

            years = list(extra_co2_damage_price_df[GlossaryCore.Years].values.tolist())

            chart_name = GlossaryCore.ExtraCO2tDamagePrice

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, f'Price ({GlossaryCore.ExtraCO2tDamagePriceDf["unit"]})', chart_name=chart_name,
                                                 y_min_zero=True)

            visible_line = True

            # add CO2 damage price serie
            new_series = InstanciatedSeries(
                years, extra_co2_damage_price.tolist(), 'CO2 damage price', 'lines', visible_line)
            new_chart.add_series(new_series)

            # add chart
            instanciated_charts.append(new_chart)

        def damage_fraction(damage):
            return damage / (1 + damage) * 100

        if True:

            tipping_point_model = self.get_sosdisc_inputs('tipping_point')
            tp_a1 = self.get_sosdisc_inputs("tp_a1")
            tp_a2 = self.get_sosdisc_inputs("tp_a2")
            tp_a3 = self.get_sosdisc_inputs("tp_a3")
            tp_a4 = self.get_sosdisc_inputs("tp_a4")

            def damage_function_tipping_point_weitzmann(temp_increase):
                return (temp_increase / tp_a1) ** tp_a2 + (temp_increase / tp_a3) ** tp_a4

            temperature_increase = np.linspace(0, 8, 100)

            damage_frac = damage_fraction(damage_function_tipping_point_weitzmann(temperature_increase))

            chart_name = "Damages models"
            new_chart = TwoAxesInstanciatedChart('Temperature increase (°C)',
                                                 'Impact on GDP (%)',
                                                 chart_name=chart_name)

            legend = "Tipping point damage model (Weitzman, 2009)" + ' (selected model)'* tipping_point_model
            new_series = InstanciatedSeries(list(temperature_increase), list(damage_frac),
                legend, 'lines', True)

            new_chart.add_series(new_series)

            damag_int = self.get_sosdisc_inputs("damag_int")
            damag_quad = self.get_sosdisc_inputs("damag_quad")

            def damage_function_tipping_point_nordhaus(temp_increase):
                return damag_int * temp_increase + damag_quad * temp_increase ** 2

            temperature_increase = np.linspace(0, 8, 100)

            damage_frac = damage_fraction(damage_function_tipping_point_nordhaus(temperature_increase))

            legend = "Standard DICE damage model (Nordhaus, 2017)" + " (selected model)" * (not tipping_point_model)
            new_series = InstanciatedSeries(list(temperature_increase), list(damage_frac),
                legend, 'lines', True)

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
