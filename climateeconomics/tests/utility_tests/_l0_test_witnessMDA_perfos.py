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
import csv

import pandas as pd
import seaborn as sns


'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import cProfile
import os
import platform
import pstats
import unittest
from io import StringIO
from os.path import dirname, join
from pathlib import Path
from shutil import rmtree
from time import sleep

import matplotlib.pyplot as plt

from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import Study
from climateeconomics.sos_processes.iam.witness.witness_coarse.usecase_witness_coarse_new import Study as Studycoarse
from climateeconomics.sos_processes.iam.witness.witness_optim_process.usecase_witness_optim import Study as StudyMDO
from climateeconomics.sos_processes.iam.witness.witness_dev_ms_process._usecase_witness_dev_ms import Study as Witness_ms_study
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


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
        input_dict_to_load[f'{self.name}.sub_mda_class'] = 'GSPureNewtonMDA'
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
#         with open(join(dirname(__file__), f'witness_perfos.csv'), 'w+') as f:
#             # f = open(result.rsplit('.')[0] + '.csv', 'w')
#             f.write(result)
#             f.close()

        lines = result.split('\n')
        total_time = float(lines[1].split(',')[3])
        print('total_time : ', total_time)

        print('filename(function),total time, time per call, number of calls')
        for line in lines[1:200]:
            print(line.split(',')[-1].split('\\')[-3:], ',', line.split(',')
                  [3], ',', line.split(',')[4], ',', line.split(',')[0])

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
        gauss_seidel_time = float([line for line in lines if 'gauss_seidel.py' in line][0].split(',')[
            3])

        _convert_array_into_new_type = float(
            [line for line in lines if '_convert_array_into_new_type' in line][0].split(',')[
                3])
        print('_convert_array_into_new_type : ', _convert_array_into_new_type)
        labels = 'Linearize', 'Pre-run', 'Gauss Seidel', 'Execute', 'Matrix Inversion', 'Matrix Build', 'Others'
        sizes = [linearize_time, pre_run_mda_time, gauss_seidel_time, execute_time, inversion_time, dres_dvar_time,
                 total_time - linearize_time - execute_time - inversion_time - dres_dvar_time - pre_run_mda_time - gauss_seidel_time]

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = pct * total / 100.0
                return '{p:.2f}%  ({v:.1f}s)'.format(p=pct, v=val)

            return my_autopct

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct=make_autopct(sizes),
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        ax1.set_title(
            f"WITNESS {mda_class} cache with {n_processes} procs, Total time : {total_time} s")

        fig_name = f'WITNESS_{mda_class}_{n_processes}_proc.png'
        plt.savefig(
            join(dirname(__file__), fig_name))
        if platform.system() == 'Windows':
            plt.show()
        else:

            #
            os.system(
                f'git add ./climateeconomics/tests/utility_tests/{fig_name}')
            os.system(
                f'git add ./climateeconomics/tests/utility_tests/witness_perfos.csv')
            os.system(
                f'git commit -m "Add {fig_name}"')

            os.system('git pull')
            os.system('git push')

    def test_02_witness_perfos_execute_GSNR(self):

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
        with open(join(dirname(__file__), 'witness_perfos.csv'), 'w+') as f:
            # f = open(result.rsplit('.')[0] + '.csv', 'w')
            f.write(result)
            f.close()

        lines = result.split('\n')
        total_time = float(lines[1].split(',')[3])
        print('total_time : ', total_time)
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
        gauss_seidel_time = float([line for line in lines if 'gauss_seidel.py' in line][0].split(',')[
            3])

        _convert_array_into_new_type = float(
            [line for line in lines if '_convert_array_into_new_type' in line][0].split(',')[
                3])
        print('_convert_array_into_new_type : ', _convert_array_into_new_type)
        labels = 'Linearize', 'Pre-run', 'Gauss Seidel', 'Execute', 'Matrix Inversion', 'Matrix Build', 'Others'
        sizes = [linearize_time, pre_run_mda_time, gauss_seidel_time, execute_time, inversion_time, dres_dvar_time,
                 total_time - linearize_time - execute_time - inversion_time - dres_dvar_time - pre_run_mda_time - gauss_seidel_time]

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = pct * total / 100.0
                return '{p:.2f}%  ({v:.1f}s)'.format(p=pct, v=val)

            return my_autopct

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct=make_autopct(sizes),
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        ax1.set_title(
            f"WITNESS {mda_class} cache with {n_processes} procs, Total time : {total_time} s")

        fig_name = f'WITNESS_{mda_class}_{n_processes}_proc.png'
        plt.savefig(
            join(dirname(__file__), fig_name))
        if platform.system() == 'Windows':
            plt.show()
        else:

            os.system(
                f'git add ./climateeconomics/tests/utility_tests/{fig_name}')
            #             os.system(
            # f'git add
            # ./climateeconomics/tests/utility_tests/witness_perfos.csv')
            os.system(
                f'git commit -m "Add {fig_name}"')

    #             os.system('git pull')
    #             os.system('git push')

    def test_03_witness_perfos_execute_PureParallel(self):

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
        input_dict_to_load[f'{self.name}.sub_mda_class'] = 'GSPureNewtonMDA'
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
        with open(join(dirname(__file__), 'witness_perfos.csv'), 'w+') as f:
            # f = open(result.rsplit('.')[0] + '.csv', 'w')
            f.write(result)
            f.close()

        lines = result.split('\n')
        total_time = float(lines[1].split(',')[3])
        print('total_time : ', total_time)
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
        gauss_seidel_time = float([line for line in lines if 'gauss_seidel.py' in line][0].split(',')[
            3])

        _convert_array_into_new_type = float(
            [line for line in lines if '_convert_array_into_new_type' in line][0].split(',')[
                3])
        print('_convert_array_into_new_type : ', _convert_array_into_new_type)
        labels = 'Linearize', 'Pre-run', 'Gauss Seidel', 'Execute', 'Matrix Inversion', 'Matrix Build', 'Others'
        sizes = [linearize_time, pre_run_mda_time, gauss_seidel_time, execute_time, inversion_time, dres_dvar_time,
                 total_time - linearize_time - execute_time - inversion_time - dres_dvar_time - pre_run_mda_time - gauss_seidel_time]

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = pct * total / 100.0
                return '{p:.2f}%  ({v:.1f}s)'.format(p=pct, v=val)

            return my_autopct

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct=make_autopct(sizes),
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        ax1.set_title(
            f"WITNESS {mda_class} cache with {n_processes} procs, Total time : {total_time} s")

        fig_name = f'WITNESS_{mda_class}_{n_processes}_proc.png'
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

    def test_04_witness_coarseperfos_multiproc(self):

        linearize_time_list = []
        execute_time_list = []
        total_time_list = []
        n_proc_list = [1, 2, 4, 8, 16]
        for n_proc in n_proc_list:
            self.name = 'Test'
            self.ee = ExecutionEngine(self.name)
            repo = 'climateeconomics.sos_processes.iam.witness'
            builder = self.ee.factory.get_builder_from_process(
                repo, 'witness_coarse')

            self.ee.factory.set_builders_to_coupling_builder(builder)
            self.ee.configure()
            usecase = Studycoarse(execution_engine=self.ee)
            usecase.study_name = self.name
            values_dict = usecase.setup_usecase()

            input_dict_to_load = {}

            for uc_d in values_dict:
                input_dict_to_load.update(uc_d)

            input_dict_to_load[f'{self.name}.n_processes'] = n_proc
            input_dict_to_load[f'{self.name}.max_mda_iter'] = 300
            input_dict_to_load[f'{self.name}.sub_mda_class'] = 'GSPureNewtonMDA'
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
            with open(join(dirname(__file__), f'witness_coarse_perfos{n_processes}.csv'), 'w+') as f:
                # f = open(result.rsplit('.')[0] + '.csv', 'w')
                f.write(result)
                f.close()

            lines = result.split('\n')
            total_time = float(lines[1].split(',')[3])
            print('total_time : ', total_time)
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
            gauss_seidel_time = float([line for line in lines if 'gauss_seidel.py' in line][0].split(',')[
                3])

            _convert_array_into_new_type = float(
                [line for line in lines if '_convert_array_into_new_type' in line][0].split(',')[
                    3])
            print('_convert_array_into_new_type : ',
                  _convert_array_into_new_type)
            labels = 'Linearize', 'Pre-run', 'Gauss Seidel', 'Execute', 'Matrix Inversion', 'Matrix Build', 'Others'
            sizes = [linearize_time, pre_run_mda_time, gauss_seidel_time, execute_time, inversion_time, dres_dvar_time,
                     total_time - linearize_time - execute_time - inversion_time - dres_dvar_time - pre_run_mda_time - gauss_seidel_time]

            def make_autopct(values):
                def my_autopct(pct):
                    total = sum(values)
                    val = pct * total / 100.0
                    return '{p:.2f}%  ({v:.1f}s)'.format(p=pct, v=val)

                return my_autopct

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct=make_autopct(sizes),
                    shadow=True, startangle=90)
            # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.axis('equal')
            ax1.set_title(
                f"WITNESS {mda_class} cache with {n_processes} procs, Total time : {total_time} s")

            fig_name = f'WITNESScoarse_{mda_class}_{n_processes}_proc.png'
            plt.savefig(
                join(dirname(__file__), fig_name))

            linearize_time_list.append(linearize_time)
            execute_time_list.append(execute_time)
            total_time_list.append(total_time)

        fig = plt.figure()
        plt.plot(n_proc_list, total_time_list, label='Total time')
        plt.plot(n_proc_list, execute_time_list, label='Execute time')
        plt.plot(n_proc_list, linearize_time_list, label='Linearize time')
        plt.legend()
        fig_name = f'WITNESScoarse_{mda_class}_allprocs.png'
        plt.savefig(
            join(dirname(__file__), fig_name))

    def test_05_witness_perfos_multiproc(self):

        linearize_time_list = []
        execute_time_list = []
        total_time_list = []
        n_proc_list = [1, 2, 4, 8, 16, 32, 64]
        for n_proc in n_proc_list:
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

            input_dict_to_load[f'{self.name}.n_processes'] = n_proc
            input_dict_to_load[f'{self.name}.max_mda_iter'] = 300
            input_dict_to_load[f'{self.name}.sub_mda_class'] = 'GSPureNewtonMDA'
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
            with open(join(dirname(__file__), f'witness_coarse_perfos{n_processes}.csv'), 'w+') as f:
                # f = open(result.rsplit('.')[0] + '.csv', 'w')
                f.write(result)
                f.close()

            lines = result.split('\n')
            total_time = float(lines[1].split(',')[3])
            print('total_time : ', total_time)
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
            gauss_seidel_time = float([line for line in lines if 'gauss_seidel.py' in line][0].split(',')[
                3])

            _convert_array_into_new_type = float(
                [line for line in lines if '_convert_array_into_new_type' in line][0].split(',')[
                    3])
            print('_convert_array_into_new_type : ',
                  _convert_array_into_new_type)
            labels = 'Linearize', 'Pre-run', 'Gauss Seidel', 'Execute', 'Matrix Inversion', 'Matrix Build', 'Others'
            sizes = [linearize_time, pre_run_mda_time, gauss_seidel_time, execute_time, inversion_time, dres_dvar_time,
                     total_time - linearize_time - execute_time - inversion_time - dres_dvar_time - pre_run_mda_time - gauss_seidel_time]

            def make_autopct(values):
                def my_autopct(pct):
                    total = sum(values)
                    val = pct * total / 100.0
                    return '{p:.2f}%  ({v:.1f}s)'.format(p=pct, v=val)

                return my_autopct

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct=make_autopct(sizes),
                    shadow=True, startangle=90)
            # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.axis('equal')
            ax1.set_title(
                f"WITNESS {mda_class} cache with {n_processes} procs, Total time : {total_time} s")

            fig_name = f'WITNESS_{mda_class}_{n_processes}_proc.png'
            plt.savefig(
                join(dirname(__file__), fig_name))

            linearize_time_list.append(linearize_time)
            execute_time_list.append(execute_time)
            total_time_list.append(total_time)
            if not platform.system() == 'Windows':
                os.system(
                    f'git add ./climateeconomics/tests/utility_tests/{fig_name}')

        fig = plt.figure()
        plt.plot(n_proc_list, total_time_list, label='Total time')
        plt.plot(n_proc_list, execute_time_list, label='Execute time')
        plt.plot(n_proc_list, linearize_time_list, label='Linearize time')
        plt.legend()
        fig_name = f'WITNESS_{mda_class}_allprocs.png'
        plt.savefig(
            join(dirname(__file__), fig_name))

        if platform.system() == 'Windows':
            plt.show()
        else:

            os.system(
                f'git add ./climateeconomics/tests/utility_tests/{fig_name}')
            os.system(
                f'git commit -m "Add {fig_name}"')
            os.system('git pull')
            os.system('git push')

    def _test_06_witness_perfos_parallel_comparison(self):

        def execute_full_usecase(n_processes):

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

            input_dict_to_load[f'{self.name}.n_processes'] = n_processes
            input_dict_to_load[f'{self.name}.max_mda_iter'] = 300
            input_dict_to_load[f'{self.name}.sub_mda_class'] = 'GSPureNewtonMDA'
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

            lines = result.split('\n')
            total_time = float(lines[1].split(',')[3])
            print('total_time : ', total_time)
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
            gauss_seidel_time = float([line for line in lines if 'gauss_seidel.py' in line][0].split(',')[
                3])

            _convert_array_into_new_type = float(
                [line for line in lines if '_convert_array_into_new_type' in line][0].split(',')[
                    3])
            sizes = [linearize_time, pre_run_mda_time, gauss_seidel_time, execute_time, inversion_time, dres_dvar_time,
                     _convert_array_into_new_type,
                     total_time - linearize_time - execute_time - inversion_time - dres_dvar_time - pre_run_mda_time - gauss_seidel_time]

            return sizes

        def stack_bar_chart(data, x, title, figsize, color, xlabel, ylabel, tailles, trick=False,
                            given_labels=False):
            """This function builds a barplot, displays the percentage of each hue(subcategory)
                and orders the bars according to a given order
            """
            plt.figure(figsize=figsize)
            ax = data.plot(
                x=x,
                kind='bar',
                stacked=True,
                title=title,
                mark_right=True, figsize=figsize, color=color)

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
            u = 0
            for c in ax.containers:
                # custom label calculates percent and add an empty string so 0
                # value bars don't have a number
                labels = [f'{(v.get_height()) / (tailles[k]) * 100:0.1f}%' if (v.get_height()) > 0 else '' for k, v
                          in enumerate(c)]
                position = 'center'
                if (trick and (u == 3)):
                    position = 'edge'
                if (given_labels):
                    labels = [
                        f'{round(v.get_height(), 2)}%' for k, v in enumerate(c)]

                ax.bar_label(c, labels=labels, label_type=position)
                u = u + 1

                # ax.bar_label(c,labels=labels, label_type='center')

            plt.savefig('witness_coarse_parallel_perfos.jpg')
            plt.show()

        def witness_parallel_status():
            pd.set_option('display.max_columns', None)
            execution_modes = ['sequential', 'threading_10_cores', 'threading_20_cores', 'multiprocessing_10_cores',
                               'multiprocessing_20_cores']
            linearizes_coarse = [41, 38, 38, 30, 33]
            execute_coarse = [24, 23, 23, 35, 35]
            others_coarse = [20, 20, 20, 22, 22]
            total_coarse = [94, 82, 81, 87, 90]
            total_coarse_analysis = pd.DataFrame(
                {'exec_mode': execution_modes, 'run_time': total_coarse})

            linearizes_full = [327, 345, 349, 107, 135]
            execute_full = [73, 66, 70, 108, 117]
            others_full = [144, 137, 136, 165, 165]
            total_full = [543, 548, 555, 380, 417]
            execution_modes_new = ['sequential', 'threading_10_cores', 'threading_20_cores', 'multiprocessing_10_cores',
                                   'multiprocessing_20_cores', 'multiproc_10_seq', 'multiproc_20_seq']
            total_full_new = [543, 548, 555, 380, 417, 327, 346]
            total_full_analysis = pd.DataFrame(
                {'exec_mode': execution_modes_new, 'run_time': total_full_new})
            coarse_analysis = pd.DataFrame(
                {'exec_mode': execution_modes, 'Linearize': linearizes_coarse, 'Execute': execute_coarse,
                 'Others': others_coarse})
            full_analysis = pd.DataFrame(
                {'exec_mode': execution_modes, 'Linearize': linearizes_full, 'Execute': execute_full,
                 'Others': others_full})
            # stack_bar_chart(data=full_analysis, x='exec_mode', title="witness full parallel perfo", figsize=(12, 10),
            #                 color=["green", "red", "gold"], xlabel="execution_mode", ylabel="running_time",
            #                 tailles=total_full)
            # stack_bar_chart(data=coarse_analysis, x='exec_mode', title="witness coarse parallel perfo", figsize=(12, 10),
            #                 color=["green", "red", "gold"], xlabel="execution_mode", ylabel="running_time",
            #                 tailles=total_coarse)

            sns.set_style('darkgrid')
            plt.style.use('ggplot')
            plt.figure(figsize=(12, 10))
            ax = sns.barplot(x='exec_mode', y='run_time',
                             data=total_coarse_analysis, ec='k')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
            ax.set_title("total run time witness coarse")
            for c in ax.containers:
                for k, v in enumerate(c):
                    x = v.get_x() + v.get_width() / 2
                    y = v.get_height()
                    if y < 60:
                        text = round(y)
                    else:
                        text = f'{round(y//60)} mn {round(y%60)} s'
                    ax.text(x, y, text, ha='center')
            plt.savefig('total_time_coarse.jpg')
            plt.show()

        def plot_hbar(file_name):
            disciplines_list = []
            disciplines_run_time = []
            with open(file_name) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        for disc in row:
                            disciplines_list.append(disc)
                    else:
                        for time in row:
                            disciplines_run_time.append(round(float(time), 2))
                    line_count += 1
            df = pd.DataFrame({'disciplines': disciplines_list,
                               'run_time': disciplines_run_time})
            df = df.sort_values('run_time', ascending=False)
            df = df.head(30)
            disciplines_run_time.sort(reverse=True)
            total_run_time = sum(disciplines_run_time)
            sns.set_style('darkgrid')
            plt.style.use('ggplot')
            plt.figure(figsize=(12, 10))
            ax = sns.barplot(x='run_time', y='disciplines', data=df, ec='k')
            ax.set_title("execution time per discipline")
            for c in ax.containers:
                for k, v in enumerate(c):
                    x = disciplines_run_time[k] + 0.005
                    y = v.get_y() + v.get_width() + 0.005
                    text = f'{round ((disciplines_run_time[k]/total_run_time)*100)} %'
                    ax.text(x, y, text)
            plt.savefig('execution_time_per_disciplines_coarse_percent.jpg')
            plt.show()

        labels = ['n_processes', 'Linearize', 'Pre-run', 'Gauss Seidel', 'Execute', 'Matrix Inversion', 'Matrix Build',
                  'convert_array_into_new_types', 'Others']
        n_processes_list = [1, 10, 20, 30]
        n_processes_list = [1]

        # dict_perfo = {}
        # for n_core in n_processes_list:
        #     performances_list = []
        #     for i in range(0, 1):
        #         performances_list.append(execute_full_usecase(n_core))
        #     dict_perfo[n_core] = [sum(x) / 1 for x in zip(*performances_list)]

        # with open(join(dirname(__file__), 'witness_full_parallel_perfos_processing_sequential.csv'), 'w+') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(labels)
        #     for n_core in n_processes_list:
        #         data = [f'with_{n_core}_cores']
        #         data.extend(dict_perfo[n_core])
        #         writer.writerow(data)
        # witness_parallel_status()
        plot_hbar('execute_split_coarse.csv')
        print("done")

    def test_07_witness_memory_perfos(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        repo = 'climateeconomics.sos_processes.iam.witness'
        builder = self.ee.factory.get_builder_from_process(
            repo, 'witness')

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        usecase = Study(execution_engine=self.ee)
        # usecase = Study(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        input_dict_to_load = {}

        for uc_d in values_dict:
            input_dict_to_load.update(uc_d)

        input_dict_to_load[f'{self.name}.WITNESS_MDO.max_iter'] = 5
        # input_dict_to_load[f'{self.name}.WITNESS_MDO.WITNESS_Eval.max_mda_iter'] = 5

        input_dict_to_load[f'{self.name}.max_mda_iter'] = 300
        input_dict_to_load[f'{self.name}.sub_mda_class'] = 'GSPureNewtonMDA'
        self.ee.load_study_from_input_dict(input_dict_to_load)

        import tracemalloc
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        self.ee.execute()
        snapshot2 = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        print(
            f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()

    # def test_08_mda_coupling_perfos_configure(self):
    #
    #     self.name = 'Test'
    #     self.ee = ExecutionEngine(self.name)
    #     repo = 'climateeconomics.sos_processes.iam.witness'
    #     builder = self.ee.factory.get_builder_from_process(
    #         repo, 'witness_dev_ms_process')
    #
    #     self.ee.factory.set_builders_to_coupling_builder(builder)
    #     self.ee.configure()
    #
    #     usecase = Witness_ms_study(execution_engine=self.ee)
    #     usecase.study_name = self.name
    #     values_dict = usecase.setup_usecase()
    #     self.ee.load_study_from_input_dict(values_dict)
    #
    #     profil = cProfile.Profile()
    #     profil.enable()
    #     self.ee.configure()
    #     profil.disable()
    #
    #     result = StringIO()
    #
    #     nb_var = len(self.ee.dm.data_id_map)
    #     print('Total Var nbr:', nb_var)
    #
    #     ps = pstats.Stats(profil, stream=result)
    #     ps.sort_stats('cumulative')
    #     ps.print_stats(1000)
    #     result = result.getvalue()
    #     # chop the string into a csv-like buffer
    #     result = 'ncalls' + result.split('ncalls')[-1]
    #     result = '\n'.join([','.join(line.rstrip().split(None, 5))
    #                         for line in result.split('\n')])
    #
    #     with open(join(dirname(__file__), f'configure_perfos.csv'), 'w+') as f:
    #         # f = open(result.rsplit('.')[0] + '.csv', 'w')
    #         f.write(result)
    #         f.close()
    #
    #     df = pd.read_csv(join(dirname(__file__), f'configure_perfos.csv'))
    #     df['Type'] = 'other'
    #     df.loc[df['filename:lineno(function)'].str.contains('mda.py'), 'Type'] = 'mda'
    #     df.loc[df['filename:lineno(function)'].str.contains('dependency_graph.py'), 'Type'] = 'dependency_graph'
    #     df.loc[df['filename:lineno(function)'].str.contains('coupling_structure'), 'Type'] = 'coupling_structure'
    #     df.loc[df['filename:lineno(function)'].str.contains('__create_graph'), 'Type'] = '__create_graph'
    #     df.loc[df['filename:lineno(function)'].str.contains('__create_condensed_graph'), 'Type'] = '__create_condensed_graph'
    #     df.loc[df['filename:lineno(function)'].str.contains('get_disciplines_couplings'), 'Type'] = 'get_disciplines_couplings'
    #     df.loc[df['filename:lineno(function)'].str.contains('__get_leaves'), 'Type'] = '__get_leaves'
    #     df.loc[df['filename:lineno(function)'].str.contains('__get_ordered_scc'), 'Type'] = '__get_ordered_scc'
    #     df.loc[df['filename:lineno(function)'].str.contains('get_execution_sequence'), 'Type'] = 'get_execution_sequence'
    #     df.loc[df['filename:lineno(function)'].str.contains('strongly_coupled_disciplines'), 'Type'] = 'strongly_coupled_disciplines'
    #     df.loc[df['filename:lineno(function)'].str.contains('weakly_coupled_disciplines'), 'Type'] = 'weakly_coupled_disciplines'
    #     df.loc[df['filename:lineno(function)'].str.contains('strong_couplings'), 'Type'] = 'strong_couplings'
    #     df.loc[df['filename:lineno(function)'].str.contains('weak_couplings'), 'Type'] = 'weak_couplings'
    #     df.loc[df['filename:lineno(function)'].str.contains('get_all_couplings'), 'Type'] = 'get_all_couplings'
    #     df.loc[df['filename:lineno(function)'].str.contains('find_discipline'), 'Type'] = 'find_discipline'
    #
    #     total_time = df['tottime'].sum()
    #     grp = df.groupby('Type').sum().reset_index()
    #     grp['ratio'] = grp['tottime']/total_time
    #
    #     print(grp[['Type', 'tottime', 'ratio']])


if '__main__' == __name__:
    cls = TestScatter()
    cls.test_01_witness_perfos_execute()
