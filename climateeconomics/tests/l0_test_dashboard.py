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

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump

from climateeconomics.sos_processes.iam.witness.witness_coarse_dev.usecase_witness_coarse_new import (
    Study,
)


class PostProcessEnergy(unittest.TestCase):
    def setUp(self):
        """
        setup of test
        """
        self.study_name = 'post-processing'
        self.repo = 'climateeconomics.sos_processes.iam.witness'
        self.proc_name = 'witness_coarse_dev'

        self.ee = ExecutionEngine(self.study_name)
        builder = self.ee.factory.get_builder_from_process(repo=self.repo,
                                                           mod_id=self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        self.usecase = Study()
        self.usecase.study_name = self.study_name
        values_dict = self.usecase.setup_usecase()
        for values_dict_i in values_dict:
            self.ee.load_study_from_input_dict(values_dict_i)

        self.namespace_list = [self.ee.ns_manager.get_all_namespace_with_name('ns_dashboard')[0].value]

    def test_post_processing_Table_plots(self):
        """
        Test to compare WITNESS energy capex, opex, CO2 tax prices
        tables for each energy / each techno per energy
        """
        from os.path import dirname, exists, join

        dump_dir = join(dirname(__file__), 'data', self.ee.study_name)
        if exists(dump_dir):
            BaseStudyManager.static_load_data(
                dump_dir, self.ee, DirectLoadDump())
        else:
            self.ee.execute()
            BaseStudyManager.static_dump_data(
                dump_dir, self.ee, DirectLoadDump())

        ppf = PostProcessingFactory()

        for itm in self.namespace_list:
            filters = ppf.get_post_processing_filters_by_namespace(self.ee, itm)
            graph_list = ppf.get_post_processing_by_namespace(self.ee, itm, filters,
                                                              as_json=False)

            for graph in graph_list:
                # graph.to_plotly().show()
                pass


if '__main__' == __name__:
    cls = PostProcessEnergy()
    cls.setUp()
    cls.test_post_processing_Table_plots()
