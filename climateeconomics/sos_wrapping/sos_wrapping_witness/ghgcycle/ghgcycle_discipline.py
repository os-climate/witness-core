'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/07-2023/11/03 Copyright 2023 Capgemini

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

# coding: utf-8
from climateeconomics.core.core_witness.climateeco_discipline import (
    ClimateEcoDiscipline,
)
from climateeconomics.core.core_witness.ghg_cycle_model import GHGCycle
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class GHGCycleDiscipline(ClimateEcoDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'Greenhouse Gas Cycle WITNESS Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-recycle fa-fw',
        'version': '',
    }
    _maturity = 'Research'

    years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1)

    # init concentrations in each box from FUND repo in ppm/volume in 1950
    # https://github.com/fund-model/MimiFUND.jl/blob/master/src
    co2_init_conc_fund = np.array([296.002949511, 5.52417779186, 6.65150094285, 2.39635475726, 0.17501699667]) * 412.4/296.002949511

    DESC_IN = {
        GlossaryCore.YearStart: ClimateEcoDiscipline.YEAR_START_DESC_IN,
        GlossaryCore.YearEnd: GlossaryCore.YearEndVar,
        GlossaryCore.TimeStep: ClimateEcoDiscipline.TIMESTEP_DESC_IN,
        GlossaryCore.GHGEmissionsDfValue: GlossaryCore.GHGEmissionsDf,
        'co2_emissions_fractions': {'type': 'list', 'subtype_descriptor': {'list': 'float'}, 'unit': '-', 'default': [0.13, 0.20, 0.32, 0.25, 0.10], 'user_level': 2},
        'co2_boxes_decays': {'type': 'list', 'subtype_descriptor': {'list': 'float'}, 'unit': GlossaryCore.Years,
                             'default': [1.0, 0.9972489701005488, 0.9865773841008381, 0.942873143854875, 0.6065306597126334],
                             'user_level': 2},
        'co2_boxes_init_conc': {'type': 'array', 'unit': 'ppm', 'default': co2_init_conc_fund, 'user_level': 2},
        'co2_pre_indus_conc': {'type': 'float', 'unit': 'ppm', 'default': DatabaseWitnessCore.CO2PreIndustrialConcentration.value, 'user_level': 2},
        'ch4_decay_rate': {'type': 'float', 'unit': 'ppb/year', 'default': 1/12, 'user_level': 2},
        'ch4_pre_indus_conc': {'type': 'float', 'unit': 'ppb', 'default': DatabaseWitnessCore.CH4PreIndustrialConcentration.value, 'user_level': 2},
        'ch4_init_conc': {'type': 'float', 'unit': 'ppb', 'user_level': 2},
        'n2o_decay_rate': {'type': 'float', 'unit': 'ppb/year', 'default':  1/114, 'user_level': 2},
        'n2o_pre_indus_conc': {'type': 'float', 'unit': 'ppb', 'default': DatabaseWitnessCore.N2OPreIndustrialConcentration.value, 'user_level': 2},
        'n2o_init_conc': {'type': 'float', 'unit': 'ppb', 'user_level': 2},
        'rockstrom_constraint_ref': {'type': 'float', 'unit': 'ppm', 'default': 490, 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE},
        'minimum_ppm_limit': {'type': 'float', 'unit': 'ppm', 'default': 250, 'user_level': 2},
        'minimum_ppm_constraint_ref': {'type': 'float', 'unit': 'ppm', 'default': 10, 'user_level': 2, 'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY, 'namespace': GlossaryCore.NS_REFERENCE},
        'GHG_global_warming_potential20': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'},
                                           'unit': 'kgCO2eq/kg',
                                           'default': ClimateEcoDiscipline.GWP_20_default,
                                           'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                           'namespace': GlossaryCore.NS_WITNESS, 'user_level': 3},
        'GHG_global_warming_potential100': {'type': 'dict', 'subtype_descriptor': {'dict': 'float'},
                                            'unit': 'kgCO2eq/kg',
                                            'default': ClimateEcoDiscipline.GWP_100_default,
                                            'visibility': ClimateEcoDiscipline.SHARED_VISIBILITY,
                                            'namespace': GlossaryCore.NS_WITNESS, 'user_level': 3},
        GlossaryCore.CheckRangeBeforeRunBoolName: GlossaryCore.CheckRangeBeforeRunBool,

    }

    DESC_OUT = {
        GlossaryCore.GHGCycleDfValue: {'type': 'dataframe', 'unit': '-', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        'ghg_cycle_df_detailed': {'type': 'dataframe', 'unit': 'ppm', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS},
        'gwp20_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': '-'},
        'gwp100_objective': {'type': 'array', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': '-'},
        'rockstrom_limit_constraint': {'type': 'array', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': '-'},
        'minimum_ppm_constraint': {'type': 'array', 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'unit': '-'},
        GlossaryCore.ExtraCO2EqSincePreIndustrialValue: GlossaryCore.ExtraCO2EqSincePreIndustrialDf,
        GlossaryCore.ExtraCO2EqSincePreIndustrialDetailedValue: GlossaryCore.ExtraCO2EqSincePreIndustrialDetailedDf,
        GlossaryCore.GlobalWarmingPotentialdDfValue: GlossaryCore.GlobalWarmingPotentialdDf,
        'pre_indus_gwp_20': {'type': 'float', 'unit': 'GtCO2Eq'},
        'pre_indus_gwp_100': {'type': 'float', 'unit': 'GtCO2Eq'},
    }

    def setup_sos_disciplines(self):
        self.update_default_values()

    def update_default_values(self):
        disc_in = self.get_data_in()
        if disc_in is not None and GlossaryCore.YearStart in disc_in:
            year_start = self.get_sosdisc_inputs(GlossaryCore.YearStart)
            if year_start is not None:
                self.update_default_value("n2o_init_conc", 'in', DatabaseWitnessCore.HistoricN2OConcentration.get_value_at_year(year_start))
                self.update_default_value("ch4_init_conc", 'in', DatabaseWitnessCore.HistoricCH4Concentration.get_value_at_year(year_start))

    def init_execution(self):
        param_in = self.get_sosdisc_inputs()
        self.ghg_cycle = GHGCycle(param_in)

    def run(self):
        # get input of discipline
        param_in = self.get_sosdisc_inputs()
        

        # compute output
        self.ghg_cycle.compute(param_in)

        dict_values = {
            GlossaryCore.GHGCycleDfValue: self.ghg_cycle.ghg_cycle_df[[GlossaryCore.Years, GlossaryCore.CO2Concentration, GlossaryCore.CH4Concentration, GlossaryCore.N2OConcentration]],
            'ghg_cycle_df_detailed': self.ghg_cycle.ghg_cycle_df,
            'gwp20_objective': self.ghg_cycle.gwp20_obj,
            'gwp100_objective': self.ghg_cycle.gwp100_obj,
            'rockstrom_limit_constraint': self.ghg_cycle.rockstrom_limit_constraint,
            'minimum_ppm_constraint': self.ghg_cycle.minimum_ppm_constraint,
            GlossaryCore.ExtraCO2EqSincePreIndustrialValue: self.ghg_cycle.extra_co2_eq,
            GlossaryCore.ExtraCO2EqSincePreIndustrialDetailedValue: self.ghg_cycle.extra_co2_eq_detailed,
            GlossaryCore.GlobalWarmingPotentialdDfValue: self.ghg_cycle.global_warming_potential_df,
            "pre_indus_gwp_20": self.ghg_cycle.pred_indus_gwp20,
            "pre_indus_gwp_100": self.ghg_cycle.pred_indus_gwp100,
        }

        
        # store data
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        """ 
        Compute jacobian for each coupling variable 
        gradient of coupling variable to compute:
        """
        d_conc_co2= self.ghg_cycle.compute_dco2_ppm_d_emissions()
        d_conc_ch4 = self.ghg_cycle.d_conc_ch4_d_emissions()
        d_conc_n2o = self.ghg_cycle.d_conc_n2o_d_emissions()
        d_ghg_ppm_d_emissions = {
            GlossaryCore.CO2: d_conc_co2,
            GlossaryCore.CH4: d_conc_ch4,
            GlossaryCore.N2O: d_conc_n2o,
        }
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.GHGCycleDfValue, GlossaryCore.CO2Concentration),
            (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCO2Emissions),
            d_ghg_ppm_d_emissions[GlossaryCore.CO2])
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.GHGCycleDfValue, GlossaryCore.CH4Concentration),
            (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCH4Emissions),
            d_ghg_ppm_d_emissions[GlossaryCore.CH4])
        self.set_partial_derivative_for_other_types(
            (GlossaryCore.GHGCycleDfValue, GlossaryCore.N2OConcentration),
            (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalN2OEmissions),
            d_ghg_ppm_d_emissions[GlossaryCore.N2O])

        # derivative gwp20 objective
        d_gwp20_objective_d_total_co2_emissions = self.ghg_cycle.d_gwp20_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions[GlossaryCore.CO2],
            specie=GlossaryCore.CO2)
        d_gwp20_objective_d_total_ch4_emissions = self.ghg_cycle.d_gwp20_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions[GlossaryCore.CH4],
            specie=GlossaryCore.CH4)
        d_gwp20_objective_d_total_n2o_emissions = self.ghg_cycle.d_gwp20_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions[GlossaryCore.N2O],
            specie=GlossaryCore.N2O)

        self.set_partial_derivative_for_other_types(
            ('gwp20_objective',), (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCO2Emissions), d_gwp20_objective_d_total_co2_emissions)
        self.set_partial_derivative_for_other_types(
            ('gwp20_objective',), (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCH4Emissions), d_gwp20_objective_d_total_ch4_emissions)
        self.set_partial_derivative_for_other_types(
            ('gwp20_objective',), (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalN2OEmissions), d_gwp20_objective_d_total_n2o_emissions)

        # derivative gwp100 objective
        d_gwp100_objective_d_total_co2_emissions = self.ghg_cycle.d_gwp100_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions[GlossaryCore.CO2],
            specie=GlossaryCore.CO2)
        d_gwp100_objective_d_total_ch4_emissions = self.ghg_cycle.d_gwp100_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions[GlossaryCore.CH4],
            specie=GlossaryCore.CH4)
        d_gwp100_objective_d_total_n2o_emissions = self.ghg_cycle.d_gwp100_objective_d_ppm(
            d_ppm=d_ghg_ppm_d_emissions[GlossaryCore.N2O],
            specie=GlossaryCore.N2O)

        self.set_partial_derivative_for_other_types(
            ('gwp100_objective',), (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCO2Emissions),
            d_gwp100_objective_d_total_co2_emissions)
        self.set_partial_derivative_for_other_types(
            ('gwp100_objective',), (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCH4Emissions),
            d_gwp100_objective_d_total_ch4_emissions)
        self.set_partial_derivative_for_other_types(
            ('gwp100_objective',), (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalN2OEmissions),
            d_gwp100_objective_d_total_n2o_emissions)

        self.set_partial_derivative_for_other_types(
            ('rockstrom_limit_constraint',), (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCO2Emissions),
            -d_ghg_ppm_d_emissions[GlossaryCore.CO2] / self.ghg_cycle.rockstrom_constraint_ref)
        self.set_partial_derivative_for_other_types(
            ('minimum_ppm_constraint',), (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCO2Emissions),
            d_ghg_ppm_d_emissions[GlossaryCore.CO2] / self.ghg_cycle.minimum_ppm_constraint_ref)

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ExtraCO2EqSincePreIndustrialValue, GlossaryCore.ExtraCO2EqSincePreIndustrialValue,),
            (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCO2Emissions),
            self.ghg_cycle.d_total_co2_equivalent_d_conc(d_conc=d_ghg_ppm_d_emissions[GlossaryCore.CO2], specie=GlossaryCore.CO2, gwp=self.ghg_cycle.gwp_20)
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ExtraCO2EqSincePreIndustrialValue, GlossaryCore.ExtraCO2EqSincePreIndustrialValue,),
            (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalCH4Emissions),
            self.ghg_cycle.d_total_co2_equivalent_d_conc(d_conc=d_ghg_ppm_d_emissions[GlossaryCore.CH4], specie=GlossaryCore.CH4, gwp=self.ghg_cycle.gwp_20)
        )

        self.set_partial_derivative_for_other_types(
            (GlossaryCore.ExtraCO2EqSincePreIndustrialValue, GlossaryCore.ExtraCO2EqSincePreIndustrialValue,),
            (GlossaryCore.GHGEmissionsDfValue, GlossaryCore.TotalN2OEmissions),
            self.ghg_cycle.d_total_co2_equivalent_d_conc(d_conc=d_ghg_ppm_d_emissions[GlossaryCore.N2O], specie=GlossaryCore.N2O, gwp=self.ghg_cycle.gwp_20)
        )

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Atmospheric concentrations',
                      GlossaryCore.ExtraCO2EqSincePreIndustrialValue,
                      GlossaryCore.GlobalWarmingPotentialdDfValue]
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
        ghg_cycle_df = deepcopy(self.get_sosdisc_outputs('ghg_cycle_df_detailed'))
        global_warming_potential_df = self.get_sosdisc_outputs(GlossaryCore.GlobalWarmingPotentialdDfValue)

        if 'Atmospheric concentrations' in chart_list:

            ppm = ghg_cycle_df[GlossaryCore.CO2Concentration]
            years = list(ppm.index)
            chart_name = 'CO2 atmospheric concentrations [ppm]'
            year_start = years[0]
            year_end = years[len(years) - 1]
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'parts per million', chart_name=chart_name)

            visible_line = True
            ordonate_data = list(ppm)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'ppm', 'lines', visible_line)
            new_chart.add_series(new_series)

            pre_industrial_level = self.get_sosdisc_inputs('co2_pre_indus_conc')
            ordonate_data = [pre_industrial_level] * len(years)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Pre-industrial level', 'dash_lines', True)
            new_chart.add_series(new_series)

            # Rockstrom Limit

            ordonate_data = [450] * int(len(years) / 5)
            abscisse_data = np.linspace(year_start, year_end, int(len(years) / 5))
            new_series = InstanciatedSeries(abscisse_data.tolist(), ordonate_data, 'Rockstrom limit', 'scatter')

            note = {'Rockstrom limit': 'Scientifical limit of the Earth'}

            new_chart.add_series(new_series)

            # Minimum PPM constraint
            ordonate_data = [self.get_sosdisc_inputs('minimum_ppm_limit')] * int(len(years) / 5)
            abscisse_data = np.linspace(year_start, year_end, int(len(years) / 5))
            new_series = InstanciatedSeries(abscisse_data.tolist(), ordonate_data, 'Minimum ppm limit', 'scatter')
            note['Minimum ppm limit'] = 'used in constraint calculation'
            new_chart.annotation_upper_left = note

            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

            ppm = ghg_cycle_df[GlossaryCore.CH4Concentration]
            years = list(ppm.index)
            chart_name = 'CH4 atmospheric concentrations [ppb]'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'parts per billion',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(ppm)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'ppb', 'lines', visible_line)
            new_chart.add_series(new_series)

            pre_industrial_level = self.get_sosdisc_inputs('ch4_pre_indus_conc')
            ordonate_data = [pre_industrial_level] * len(years)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Pre-industrial level', 'dash_lines', True)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

            ppm = ghg_cycle_df[GlossaryCore.N2OConcentration]
            years = list(ppm.index)
            chart_name = 'N2O atmospheric concentrations [ppb]'
            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, 'parts per billion',
                                                 chart_name=chart_name)

            visible_line = True
            ordonate_data = list(ppm)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'ppb', 'lines', visible_line)
            new_chart.add_series(new_series)

            pre_industrial_level = self.get_sosdisc_inputs('n2o_pre_indus_conc')
            ordonate_data = [pre_industrial_level] * len(years)
            new_series = InstanciatedSeries(
                years, ordonate_data, 'Pre-industrial level', 'dash_lines', True)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.ExtraCO2EqSincePreIndustrialValue in chart_list:
            years = list(ghg_cycle_df[GlossaryCore.Years].values)
            chart_name = GlossaryCore.ExtraCO2EqSincePreIndustrialValue

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.ExtraCO2EqSincePreIndustrialDf['unit'],
                                                 chart_name=chart_name, y_min_zero=True)

            visible_line = True
            extra_co2_eq_df = self.get_sosdisc_outputs(GlossaryCore.ExtraCO2EqSincePreIndustrialDetailedValue)

            ordonate_data = list(extra_co2_eq_df[GlossaryCore.ExtraCO2EqSincePreIndustrial2OYbasisValue])
            new_series = InstanciatedSeries(
                years, ordonate_data, "20-year basis (applied)", 'lines', visible_line)
            new_chart.add_series(new_series)

            ordonate_data = list(extra_co2_eq_df[GlossaryCore.ExtraCO2EqSincePreIndustrial10OYbasisValue])
            new_series = InstanciatedSeries(
                years, ordonate_data, "100-year basis", 'lines', visible_line)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.GlobalWarmingPotentialdDfValue in chart_list:
            years = list(global_warming_potential_df[GlossaryCore.Years].values)
            gwp_pre_indus = self.get_sosdisc_outputs('pre_indus_gwp_20')
            chart_name = f"{GlossaryCore.GlobalWarmingPotentialdDfValue} {GlossaryCore.YearBasis20}"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.GlobalWarmingPotentialdDf['unit'],
                                                 chart_name=chart_name, stacked_bar=True)

            for ghg in GlossaryCore.GreenHouseGases:
                ordonate_data = list(global_warming_potential_df[f"{ghg} {GlossaryCore.YearBasis20}"])
                new_series = InstanciatedSeries(
                    years, ordonate_data, ghg, 'bar', visible_line)
                new_chart.add_series(new_series)

            ordonate_data = list(global_warming_potential_df[f"Total {GlossaryCore.YearBasis20}"])

            new_series = InstanciatedSeries(
                years, ordonate_data, "Total", 'lines', visible_line)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, [gwp_pre_indus] * len(years), "Pre-industrial level", 'dash_lines', visible_line)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        if GlossaryCore.GlobalWarmingPotentialdDfValue in chart_list:
            years = list(global_warming_potential_df[GlossaryCore.Years].values)
            gwp_pre_indus = self.get_sosdisc_outputs('pre_indus_gwp_100')
            chart_name = f"{GlossaryCore.GlobalWarmingPotentialdDfValue} {GlossaryCore.YearBasis100}"

            new_chart = TwoAxesInstanciatedChart(GlossaryCore.Years, GlossaryCore.GlobalWarmingPotentialdDf['unit'],
                                                 chart_name=chart_name, stacked_bar=True)

            for ghg in GlossaryCore.GreenHouseGases:
                ordonate_data = list(global_warming_potential_df[f"{ghg} {GlossaryCore.YearBasis100}"])
                new_series = InstanciatedSeries(
                    years, ordonate_data, ghg, 'bar', visible_line)
                new_chart.add_series(new_series)

            ordonate_data = list(global_warming_potential_df[f"Total {GlossaryCore.YearBasis100}"])
            new_series = InstanciatedSeries(
                years, ordonate_data, "Total", 'lines', visible_line)
            new_chart.add_series(new_series)

            new_series = InstanciatedSeries(
                years, [gwp_pre_indus] * len(years), "Pre-industrial level", 'dash_lines', visible_line)
            new_chart.add_series(new_series)

            instanciated_charts.append(new_chart)

        return instanciated_charts
