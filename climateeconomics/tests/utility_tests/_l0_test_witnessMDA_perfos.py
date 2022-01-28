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
import matplotlib.pyplot as plt
import os
import platform


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

        input_dict_to_load[f'{self.name}.n_processes'] = 1
        input_dict_to_load[f'{self.name}.max_mda_iter'] = 300
        input_dict_to_load[f'{self.name}.sub_mda_class'] = 'GSNewtonMDA'
        self.ee.load_study_from_input_dict(input_dict_to_load)
        profil = cProfile.Profile()
        profil.enable()
        self.ee.execute()
        mda_class = self.ee.dm.get_value(f'{self.name}.sub_mda_class')
        n_processes = self.ee.dm.get_value(f'{self.name}.n_processes')
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
        with open(join(dirname(__file__), 'witness_coarse_perfos.csv'), 'w+') as f:
            #f = open(result.rsplit('.')[0] + '.csv', 'w')
            f.write(result)
            f.close()

        lines = result.split('\n')
        total_time = float(lines[1].split(',')[3])
        print(total_time)
        linearize_time = float([line for line in lines if 'linearize' in line][0].split(',')[
            3])
        execute_time = float([line for line in lines if 'execute_all_disciplines' in line][0].split(',')[
            3])
        inversion_time = float([line for line in lines if 'algo_lib.py' in line][0].split(',')[
            3])
        pre_run_mda_time = float([line for line in lines if 'pre_run_mda' in line][0].split(',')[
            3])
        dres_dvar_time = float([line for line in lines if 'dres_dvar' in line][0].split(',')[
            3])

        _convert_array_into_new_type = float([line for line in lines if '_convert_array_into_new_type' in line][0].split(',')[
            3])

        labels = 'Linearize', 'Pre-run', 'Execute', 'Matrix Inversion', 'Matrix Build',  'Others'
        sizes = [linearize_time, pre_run_mda_time, execute_time, inversion_time, dres_dvar_time,
                 total_time - linearize_time - execute_time - inversion_time - dres_dvar_time - pre_run_mda_time]

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = pct * total / 100.0
                return '{p:.2f}%  ({v:.1f}s)'.format(p=pct, v=val)
            return my_autopct

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes,  labels=labels, autopct=make_autopct(sizes),
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        ax1.set_title(
            f"WITNESS {mda_class} cache with {n_processes} procs, Total time : {total_time} s")

        fig_name = f'WITNESS_full_{mda_class}_{n_processes}_proc.png'
        plt.savefig(
            join(dirname(__file__), fig_name))
        if platform.system() == 'Windows':
            plt.show()
        else:

            os.system(
                f'git add ./climateeconomics/tests/utility_tests/{fig_name}')
            os.system(
                f'git add ./climateeconomics/tests/utility_tests/witness_perfos.csv')
            os.system(
                f'git commit -m "Add {fig_name}"')
            os.system('git pull')
            os.system('git push')


if '__main__' == __name__:
    cls = TestScatter()
    cls.test_01_witness_perfos_execute()
