

import unittest
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
import pandas as pd
import random as rd


class TestSoSDiscipline(unittest.TestCase):

    def setUp(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        self.factory = self.ee.factory

    def test_01_execute_process(self):

        model_name = 'CopperModel'
        ns_dict = {'ns_public': f'{self.name}'}

        self.ee.ns_manager.add_ns_def(ns_dict)
        mod_path = 'climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_copper_resource_v0.copper_disc.CopperDisc'
        builder = self.factory.get_builder_from_module(
            model_name, mod_path)

        self.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        year = 2020
        year_end = 2101

        copper_demand = pd.DataFrame(
            [(year, rd.gauss(26, 0.5), 'million_tonnes')], columns=['Year', 'Demand', 'unit'])
        extraction = [26]

        year += 1

        while year < year_end:
            copper_demand = copper_demand.append({'Year': year,
                                                  'Demand': rd.gauss(26, 0.5) * (1.056467) ** (year - 2020),
                                                  'unit': 'million_tonnes'}, ignore_index=True)
            extraction += [26 * (1.056467) ** (year - 2020)]
            year += 1

        print(extraction)

        values_dict = {'Test.CopperModel.copper_demand': copper_demand,
                       'Test.CopperModel.annual_extraction': extraction}
        self.ee.dm.set_values_from_dict(values_dict)

        self.ee.execute()

        copper_model = self.ee.dm.get_disciplines_with_name('Test.CopperModel')[
            0]
        filters = copper_model.get_chart_filter_list()
        graph_list = copper_model.get_post_processing_list(filters)
#         for graph in graph_list:
#             graph.to_plotly().show()


if __name__ == "__main__":
    unittest.main()
