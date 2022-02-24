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
from climateeconomics.sos_processes.iam.witness.witness_optim_process_independent_invest.usecase_witness_optim import Study as StudyMDO
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
            #             os.system(
            # f'git add
            # ./climateeconomics/tests/utility_tests/witness_perfos.csv')
            os.system(
                f'git commit -m "Add {fig_name}"')

    #             os.system('git pull')
    #             os.system('git push')

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

    def test_06_witness_perfos_parallel_comparison(self):

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

        labels = ['', 'Linearize', 'Pre-run', 'Gauss Seidel', 'Execute', 'Matrix Inversion', 'Matrix Build',
                  'convert_array_into_new_types', 'Others']
        n_processes_list = [1, 10, 20, 30]
        dict_perfo = {}

        for n_core in n_processes_list:
            performances_list = []
            for i in range(0, 5):
                performances_list.append(execute_full_usecase(n_core))
            dict_perfo[n_core] = [sum(x) / 5 for x in zip(*performances_list)]

        with open(join(dirname(__file__), 'witness_parallel_perfos.csv'), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(labels)
            for n_core in n_processes_list:
                data = [f'with_{n_core}_cores']
                data.extend(dict_perfo[n_core])
                writer.writerow(data)

        print("done")

    def test_07_witness_memory_perfos(self):

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)
        repo = 'climateeconomics.sos_processes.iam.witness'
        builder = self.ee.factory.get_builder_from_process(
            repo, 'witness')
        # repo, 'witness_optim_process_independent_invest')

        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        #usecase = StudyMDO(execution_engine=self.ee)
        usecase = Study(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        input_dict_to_load = {}

        for uc_d in values_dict:
            input_dict_to_load.update(uc_d)

        input_dict_to_load[f'{self.name}.WITNESS_MDO.max_iter'] = 5
        input_dict_to_load[f'{self.name}.WITNESS_MDO.WITNESS_Eval.max_mda_iter'] = 5

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
            f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()


if '__main__' == __name__:
    cls = TestScatter()
    cls.test_07_witness_memory_perfos()
