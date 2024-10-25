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
from os.path import dirname, join

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_ms_optim_ku_process.usecase_ms_ku_tipping_point import Study as StudyTP
from energy_models.models.clean_energy.clean_energy_simple_techno.clean_energy_simple_techno_disc import \
    CleanEnergySimpleTechnoDiscipline
from energy_models.models.fossil.fossil_simple_techno.fossil_simple_techno_disc import FossilSimpleTechnoDiscipline


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=2023, filename=__file__, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(filename, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.year_start = year_start
        self.study_tp = StudyTP(year_start=year_start, filename=filename, run_usecase=run_usecase, execution_engine=execution_engine)
        self.test_post_procs = True


    def setup_usecase(self, study_folder_path=None):

        values_dict = self.study_tp.setup_usecase()

        clean_tid = CleanEnergySimpleTechnoDiscipline.techno_info_dict

        clean_tid["Capex_init"] = 230.0
        clean_tid["learning_rate"] = .0
        clean_tid["Opex_percentage"] = .0
        clean_tid["resource_price"] = 70

        fossil_tid = FossilSimpleTechnoDiscipline.techno_info_dict
        fossil_tid["Opex_percentage"] = 0.024
        fossil_tid["Capex_init"] = 100.
        fossil_tid["resource_price"] = 75

        fossil_tid_varnames = [
            'usecase_ms_ku_tipping_point_old_values.optimization scenarios.- damage - tax, fossil 100%.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fossil.FossilSimpleTechno.techno_infos_dict',
            'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, fossil 40%, Tipping point 6deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fossil.FossilSimpleTechno.techno_infos_dict',
            'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, fossil 40%, Tipping point 4_5deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fossil.FossilSimpleTechno.techno_infos_dict',
            'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, fossil 40%, Tipping point 3_5deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fossil.FossilSimpleTechno.techno_infos_dict',
            'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, NZE, Tipping point 6deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fossil.FossilSimpleTechno.techno_infos_dict',
            'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, NZE, Tipping point 4_5deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fossil.FossilSimpleTechno.techno_infos_dict',
            'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, NZE, Tipping point 3_5deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.fossil.FossilSimpleTechno.techno_infos_dict']

        clean_tid_varnames = ['usecase_ms_ku_tipping_point_old_values.optimization scenarios.- damage - tax, fossil 100%.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.clean_energy.CleanEnergySimpleTechno.techno_infos_dict',
                              'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, fossil 40%, Tipping point 6deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.clean_energy.CleanEnergySimpleTechno.techno_infos_dict',
                              'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, fossil 40%, Tipping point 4_5deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.clean_energy.CleanEnergySimpleTechno.techno_infos_dict',
                              'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, fossil 40%, Tipping point 3_5deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.clean_energy.CleanEnergySimpleTechno.techno_infos_dict',
                              'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, NZE, Tipping point 6deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.clean_energy.CleanEnergySimpleTechno.techno_infos_dict',
                              'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, NZE, Tipping point 4_5deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.clean_energy.CleanEnergySimpleTechno.techno_infos_dict',
                              'usecase_ms_ku_tipping_point_old_values.optimization scenarios.+ damage + tax, NZE, Tipping point 3_5deg C.WITNESS_MDO.WITNESS_Eval.WITNESS.EnergyMix.clean_energy.CleanEnergySimpleTechno.techno_infos_dict']


        values_dict.update({
            var : clean_tid for var in clean_tid_varnames
        })
        values_dict.update({
            var : fossil_tid for var in fossil_tid_varnames
        })


        return values_dict


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()

