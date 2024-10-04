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
import pandas as pd
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.utility_tools import (
    compute_utility_objective,
    compute_utility_objective_der,
    compute_utility_quantities,
    get_inputs_for_utility_all_sectors,
    get_inputs_for_utility_per_sector,
    s_curve_function,
)
from climateeconomics.glossarycore import GlossaryCore


class SectorizedUtilityDiscipline(ClimateEcoDiscipline):
    "SectorizedUtilityDiscipline discipline"

    # ontology information
    _ontology_data = {
        "label": "Sectorized Utility WITNESS Model",
        "type": "Research",
        "source": "SoSTrades Project",
        "validated": "",
        "validated_by": "SoSTrades Project",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "fas fa-child fa-fw",
        "version": "",
    }

    SATURATION_PARAMETERS = {
        "conso_elasticity": {
            "type": "float",
            "default": 1.45,
            "unit": "-",
            "user_level": 2,
        },
        "strech_scurve": {"type": "float", "default": 1.7},
        "shift_scurve": {"type": "float", "default": -0.2},
        "init_rate_time_pref": {
            "type": "float",
            "default": 0.015,
            "unit": "-",
        },
        "initial_raw_energy_price": {
            "type": "float",
            "unit": "$/MWh",
            "default": 110,
            "user_level": 2,
        },
    }

    _maturity = "Research"

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.PopulationDfValue: GlossaryCore.PopulationDf,
        GlossaryCore.EnergyMeanPriceValue: GlossaryCore.EnergyMeanPrice,
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,
        GlossaryCore.SectorListValue: GlossaryCore.SectorList,
        GlossaryCore.SectorizedConsumptionDfValue: GlossaryCore.SectorizedConsumptionDf,
    }
    DESC_OUT = {}

    def setup_sos_disciplines(self):
        """setup dynamic inputs and outputs"""
        dynamic_inputs = {}
        dynamic_outputs = {}

        if GlossaryCore.SectorListValue in self.get_data_in():
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            for sector in sector_list:
                # Add saturation effect parameters to each sector
                for k, v in self.SATURATION_PARAMETERS.items():
                    dynamic_inputs[f"{sector}.{k}"] = v

                # Add utility per sector as output
                utility_df = deepcopy(GlossaryCore.UtilityDf)
                del utility_df['namespace'], utility_df['visibility']
                dynamic_outputs[f"{sector}.{GlossaryCore.UtilityDfValue}"] = utility_df
                dynamic_outputs[f"{sector}.{GlossaryCore.UtilityObjectiveName}"] = GlossaryCore.get_dynamic_variable(GlossaryCore.UtilityObjective)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        outputs_dict = {}
        sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
        years, population, energy_price = get_inputs_for_utility_all_sectors(self.get_sosdisc_inputs())
        for sector in sector_list:
            consumption, energy_price_ref, init_rate_time_pref, scurve_shift, scurve_stretch = get_inputs_for_utility_per_sector(
                self.get_sosdisc_inputs(), sector)

            utility_quantities = compute_utility_quantities(years, consumption, energy_price, population,
                                                            energy_price_ref, init_rate_time_pref, scurve_shift,
                                                            scurve_stretch)

            utility_df = pd.DataFrame({GlossaryCore.Years: years} | utility_quantities)
            outputs_dict[f"{sector}.{GlossaryCore.UtilityDfValue}"] = utility_df
            outputs_dict[f"{sector}.{GlossaryCore.UtilityObjectiveName}"] = np.array([
                compute_utility_objective(years, consumption, energy_price,
                                          population,
                                          energy_price_ref, init_rate_time_pref,
                                          scurve_shift,
                                          scurve_stretch)])


        self.store_sos_outputs_values(outputs_dict)

    def compute_sos_jacobian(self):
        inputs_dict = self.get_sosdisc_inputs()

        years, population, energy_price = get_inputs_for_utility_all_sectors(self.get_sosdisc_inputs())
        for sector in inputs_dict[GlossaryCore.SectorListValue]:
            consumption, energy_price_ref, init_rate_time_pref, scurve_shift, scurve_stretch = get_inputs_for_utility_per_sector(
                self.get_sosdisc_inputs(), sector)

            obj_derivatives = compute_utility_objective_der(years, consumption, energy_price, population,
                                                            energy_price_ref,
                                                            init_rate_time_pref,
                                                            scurve_shift, scurve_stretch)

            self.set_partial_derivative_for_other_types(
                (f"{sector}.{GlossaryCore.UtilityObjectiveName}",),
                (GlossaryCore.SectorizedConsumptionDfValue, sector),
                obj_derivatives[0])

            self.set_partial_derivative_for_other_types(
                (f"{sector}.{GlossaryCore.UtilityObjectiveName}",),
                (GlossaryCore.EnergyMeanPriceValue, GlossaryCore.EnergyPriceValue),
                obj_derivatives[1])

            self.set_partial_derivative_for_other_types(
                (f"{sector}.{GlossaryCore.UtilityObjectiveName}",),
                (GlossaryCore.PopulationDfValue, GlossaryCore.PopulationValue),
                obj_derivatives[2])

    def get_chart_filter_list(self):
        chart_filters = []

        chart_list = ["Sectorization"]
        chart_filters.append(ChartFilter("Charts", chart_list, chart_list, "charts"))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        chart_list = []

        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == "charts":
                    chart_list = chart_filter.selected_values

        if "Sectorization" in chart_list:
            sector_list = self.get_sosdisc_inputs(GlossaryCore.SectorListValue)
            sectors_consumption_df = self.get_sosdisc_inputs(GlossaryCore.SectorizedConsumptionDfValue)
            years = sectors_consumption_df[GlossaryCore.Years]

            # S-curve fit
            new_chart = TwoAxesInstanciatedChart(
                f'Variation of quantity of things consumed per capita since {years[0]} [%]',
                'Utility gain per capita', chart_name='Model visualisation : Quantity utility per capita function')
            for sector in sector_list:
                scurve_stretch = self.get_sosdisc_inputs(f"{sector}.strech_scurve")
                scurve_shift = self.get_sosdisc_inputs(f"{sector}.shift_scurve")
                n = 200
                ratios = np.linspace(-0.2, 4, n)
                new_series = InstanciatedSeries((ratios - 1) * 100,
                                                s_curve_function(ratios, scurve_shift, scurve_stretch),
                                                f'{sector}', 'lines', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

            # Variation of consumption
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'Variation [%]', chart_name="Variation of consumption by sector")
            for sector in sector_list:
                consumption = sectors_consumption_df[sector].to_numpy()
                variation = (consumption - consumption[0]) / consumption[0] * 100.0
                new_series = InstanciatedSeries(years, variation, f'{sector}', 'lines', True)
                new_chart.series.append(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
