'''
Copyright 2023 Capgemini

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
import unittest

import numpy as np
from pandas import DataFrame
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class EnergyInvestDiscTest(unittest.TestCase):

    def setUp(self):
        '''
        Set up function
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):
        self.model_name = 'EnergyInvest'
        ns_dict = {GlossaryCore.NS_WITNESS: f'{self.name}',
                   GlossaryCore.NS_ENERGY_MIX: f'{self.name}',
                   'ns_public': f'{self.name}',
                   GlossaryCore.NS_FUNCTIONS: f'{self.name}',
                   GlossaryCore.NS_REFERENCE: f'{self.name}',
                   'ns_energy_study': f'{self.name}',}

        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_sectors.energy_invest.energy_invest_disc.EnergyInvestDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        years = np.arange(GlossaryCore.YearStartDefault, GlossaryCore.YearEndDefault + 1, 1)
        n_years = len(years)
        time = np.arange(0, n_years)

        default_CO2_tax = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.CO2Tax: np.linspace(10, 80, n_years)}
        )

        default_co2_efficiency = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.CO2TaxEfficiencyValue: np.linspace(30, 30, n_years)}
        )

        co2_emissions_gt = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.TotalCO2Emissions: np.linspace(34, 55, n_years)})

        energy_investment_wo_tax = DataFrame(
            {GlossaryCore.Years: years,
             GlossaryCore.EnergyInvestmentsWoTaxValue:  1.02 ** time * 0.7})  # in T$

        values_dict = {
            f'{self.name}.{GlossaryCore.EnergyInvestmentsWoTaxValue}': energy_investment_wo_tax,
            f'{self.name}.{GlossaryCore.CO2TaxesValue}': default_CO2_tax,
            f'{self.name}.{self.model_name}.{GlossaryCore.CO2TaxEfficiencyValue}': default_co2_efficiency,
            f'{self.name}.{GlossaryCore.CO2EmissionsGtValue}': co2_emissions_gt,
        }

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass
