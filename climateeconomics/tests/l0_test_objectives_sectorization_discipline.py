"""
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from os.path import dirname, join

import numpy as np
from pandas import DataFrame, read_csv
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from climateeconomics.glossarycore import GlossaryCore


class ObjectivesTestCase(unittest.TestCase):

    def setUp(self):
        """
        Initialize third data needed for testing
        """
        self.year_start = 2000
        self.year_end = GlossaryCore.YearStartDefault
        nb_per = round(self.year_end - self.year_start + 1)
        self.nb_per = nb_per
        self.years = np.arange(self.year_start, self.year_end + 1)

        # Test With a GDP and capital that grows at 2%
        gdp_year_start = 130.187
        capital_year_start = 376.6387
        gdp_serie = np.zeros(self.nb_per)
        capital_serie = np.zeros(self.nb_per)
        gdp_serie[0] = gdp_year_start
        capital_serie[0] = capital_year_start
        for year in np.arange(1, self.nb_per):
            gdp_serie[year] = gdp_serie[year - 1] * 1.02
            capital_serie[year] = capital_serie[year - 1] * 1.02
        # for each sector share of total gdp 2020
        gdp_agri = gdp_serie * 6.775773 / 100
        gdp_indus = gdp_serie * 28.4336 / 100
        gdp_service = gdp_serie * 64.79 / 100
        self.prod_agri = DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.Output: gdp_agri,
                GlossaryCore.OutputNetOfDamage: gdp_agri * 0.995,
            }
        )
        self.prod_indus = DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.Output: gdp_indus,
                GlossaryCore.OutputNetOfDamage: gdp_indus * 0.995,
            }
        )
        self.prod_service = DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.Output: gdp_service,
                GlossaryCore.OutputNetOfDamage: gdp_service * 0.995,
            }
        )
        cap_agri = capital_serie * 0.018385
        cap_indus = capital_serie * 0.234987
        cap_service = capital_serie * 0.74662
        energy_eff = np.linspace(2, 3, self.nb_per)
        self.cap_agri_df = DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.Capital: cap_agri,
                GlossaryCore.UsableCapital: cap_agri * 0.8,
                GlossaryCore.EnergyEfficiency: energy_eff,
            }
        )
        self.cap_indus_df = DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.Capital: cap_indus,
                GlossaryCore.UsableCapital: cap_indus * 0.8,
                GlossaryCore.EnergyEfficiency: energy_eff,
            }
        )
        self.cap_service_df = DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.Capital: cap_service,
                GlossaryCore.UsableCapital: cap_service * 0.8,
                GlossaryCore.EnergyEfficiency: energy_eff,
            }
        )

        self.economics_df = DataFrame(
            {
                GlossaryCore.Years: self.years,
                GlossaryCore.Capital: capital_serie,
                GlossaryCore.UsableCapital: capital_serie * 0.8,
                GlossaryCore.Output: gdp_serie,
                GlossaryCore.OutputNetOfDamage: gdp_serie * 0.995,
            }
        )

        data_dir = join(dirname(__file__), "data/sectorization_fitting")
        self.hist_gdp = read_csv(join(data_dir, "hist_gdp_sect.csv"))
        self.hist_capital = read_csv(join(data_dir, "hist_capital_sect.csv"))
        self.hist_energy = read_csv(join(data_dir, "hist_energy_sect.csv"))
        long_term_energy_eff = read_csv(join(data_dir, "long_term_energy_eff_sectors.csv"))
        self.lt_enef_agri = DataFrame(
            {
                GlossaryCore.Years: long_term_energy_eff[GlossaryCore.Years],
                GlossaryCore.EnergyEfficiency: long_term_energy_eff[GlossaryCore.SectorAgriculture],
            }
        )
        self.lt_enef_indus = DataFrame(
            {
                GlossaryCore.Years: long_term_energy_eff[GlossaryCore.Years],
                GlossaryCore.EnergyEfficiency: long_term_energy_eff[GlossaryCore.SectorIndustry],
            }
        )
        self.lt_enef_services = DataFrame(
            {
                GlossaryCore.Years: long_term_energy_eff[GlossaryCore.Years],
                GlossaryCore.EnergyEfficiency: long_term_energy_eff[GlossaryCore.SectorServices],
            }
        )
        self.extra_data = read_csv(join(data_dir, "extra_data_for_energy_eff.csv"))

    def test_objectives_discipline(self):
        """
        Check discipline setup and run
        """
        name = "Test"
        model_name = "Objectives"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}",
            GlossaryCore.NS_MACRO: f"{name}.{model_name}",
            "ns_obj": f"{name}.{model_name}",
            GlossaryCore.NS_SECTORS: f"{name}.{model_name}",
        }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = (
            "climateeconomics.sos_wrapping.sos_wrapping_sectors.objectives.objectives_discipline.ObjectivesDiscipline"
        )
        builder = ee.factory.get_builder_from_module(model_name, mod_path)

        ee.factory.set_builders_to_coupling_builder(builder)

        ee.configure()
        ee.display_treeview_nodes()
        self.inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.EconomicsDfValue}": self.economics_df,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}": self.prod_agri,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.{GlossaryCore.ProductionDfValue}": self.prod_service,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ProductionDfValue}": self.prod_indus,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_indus_df,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_service_df,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_agri_df,
            f"{name}.{model_name}.historical_gdp": self.hist_gdp,
            f"{name}.{model_name}.historical_capital": self.hist_capital,
            f"{name}.{model_name}.historical_energy": self.hist_energy,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.longterm_energy_efficiency": self.lt_enef_agri,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.longterm_energy_efficiency": self.lt_enef_indus,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.longterm_energy_efficiency": self.lt_enef_services,
        }

        ee.load_study_from_input_dict(self.inputs_dict)
        ee.execute()
        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        error_pib_total = disc.get_sosdisc_outputs(["error_pib_total"])
        # print(error_pib_total, error_cap_total, sectors_cap_errors, sectors_gdp_errors)
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_objectives_discipline_withextradata(self):
        name = "Test"
        model_name = "Objectives"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}",
            GlossaryCore.NS_MACRO: f"{name}.{model_name}",
            "ns_obj": f"{name}.{model_name}",
            GlossaryCore.NS_SECTORS: f"{name}.{model_name}",
        }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = (
            "climateeconomics.sos_wrapping.sos_wrapping_sectors.objectives.objectives_discipline.ObjectivesDiscipline"
        )
        builder = ee.factory.get_builder_from_module(model_name, mod_path)
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        ee.display_treeview_nodes()

        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.EconomicsDfValue}": self.economics_df,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}": self.prod_agri,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.{GlossaryCore.ProductionDfValue}": self.prod_service,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ProductionDfValue}": self.prod_indus,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_indus_df,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_service_df,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_agri_df,
            f"{name}.{model_name}.historical_gdp": self.hist_gdp,
            f"{name}.{model_name}.historical_capital": self.hist_capital,
            f"{name}.{model_name}.historical_energy": self.hist_energy,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.longterm_energy_efficiency": self.lt_enef_agri,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.longterm_energy_efficiency": self.lt_enef_indus,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.longterm_energy_efficiency": self.lt_enef_services,
            f"{name}.{model_name}.data_for_earlier_energy_eff": self.extra_data,
        }

        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    #         for graph in graph_list:
    #             graph.to_plotly().show()

    def test_objectives_discipline_withextradata_wrongdata(self):
        name = "Test"
        model_name = "Objectives"
        ee = ExecutionEngine(name)
        ns_dict = {
            "ns_public": f"{name}",
            GlossaryCore.NS_WITNESS: f"{name}",
            GlossaryCore.NS_MACRO: f"{name}.{model_name}",
            "ns_obj": f"{name}.{model_name}",
            GlossaryCore.NS_SECTORS: f"{name}.{model_name}",
        }
        ee.ns_manager.add_ns_def(ns_dict)

        mod_path = (
            "climateeconomics.sos_wrapping.sos_wrapping_sectors.objectives.objectives_discipline.ObjectivesDiscipline"
        )
        builder = ee.factory.get_builder_from_module(model_name, mod_path)
        ee.factory.set_builders_to_coupling_builder(builder)
        ee.configure()
        ee.display_treeview_nodes()

        # modify extra data to have one column with only O
        extra_data = self.extra_data.copy()
        extra_data["Industry.capital"] = 0

        inputs_dict = {
            f"{name}.{GlossaryCore.YearStart}": self.year_start,
            f"{name}.{GlossaryCore.YearEnd}": self.year_end,
            f"{name}.{GlossaryCore.EconomicsDfValue}": self.economics_df,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.ProductionDfValue}": self.prod_agri,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.{GlossaryCore.ProductionDfValue}": self.prod_service,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.ProductionDfValue}": self.prod_indus,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_indus_df,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_service_df,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.{GlossaryCore.DetailedCapitalDfValue}": self.cap_agri_df,
            f"{name}.{model_name}.historical_gdp": self.hist_gdp,
            f"{name}.{model_name}.historical_capital": self.hist_capital,
            f"{name}.{model_name}.historical_energy": self.hist_energy,
            f"{name}.{model_name}.{GlossaryCore.SectorAgriculture}.longterm_energy_efficiency": self.lt_enef_agri,
            f"{name}.{model_name}.{GlossaryCore.SectorIndustry}.longterm_energy_efficiency": self.lt_enef_indus,
            f"{name}.{model_name}.{GlossaryCore.SectorServices}.longterm_energy_efficiency": self.lt_enef_services,
            f"{name}.{model_name}.data_for_earlier_energy_eff": extra_data,
        }

        ee.load_study_from_input_dict(inputs_dict)
        ee.execute()
        disc = ee.dm.get_disciplines_with_name(f"{name}.{model_name}")[0]
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)


#         for graph in graph_list:
#             graph.to_plotly().show()
