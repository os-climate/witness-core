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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import dirname, join
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import Study
import cProfile
import pstats
from io import StringIO


class TestScatter(unittest.TestCase):
    """
    SoSDiscipline test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'

    def tearDown(self):

        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def test_01_witness_perfos_execute(self):

        cumtime_gems_list = []
        cumtime_configure_list = []
        cumtime_build_list = []
        cumtime_treeview_list = []
        cumtime_execute_list = []
        cumtime_total_configure_list = []

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        repo = 'climateeconomics.sos_processes.iam.witness'
        builder = self.ee.factory.get_builder_from_process(
            repo, 'witness')

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        usecase = Study(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        input_dict_to_load = {}

        for uc_d in values_dict:
            input_dict_to_load.update(uc_d)

        self.ee.load_study_from_input_dict(input_dict_to_load)
        profil = cProfile.Profile()
        profil.enable()
        self.ee.execute()
        profil.disable()
        result = StringIO()

        ps = pstats.Stats(profil, stream=result)
        ps.sort_stats('cumulative')
        ps.print_stats(1000)
        result = result.getvalue()
        # chop the string into a csv-like buffer
        result = 'ncalls' + result.split('ncalls')[-1]
        result = '\n'.join([','.join(line.rstrip().split(None, 5))
                            for line in result.split('\n')])
#
        with open(join(dirname(__file__), 'witness_perfos.csv'), 'w+') as f:
            #f = open(result.rsplit('.')[0] + '.csv', 'w')
            f.write(result)
            f.close()


if '__main__' == __name__:
    cls = TestScatter()
    cls.test_01_witness_perfos_execute()
