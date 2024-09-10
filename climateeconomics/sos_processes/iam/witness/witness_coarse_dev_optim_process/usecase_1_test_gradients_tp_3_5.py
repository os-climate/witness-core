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
import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_optim_process.usecase_witness_optim_invest_distrib import (
    Study as StudyOptimInvestDistrib,
)


class Study(StudyOptimInvestDistrib):
    def __init__(self, run_usecase=False, execution_engine=None, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, time_step=1):
        super().__init__(year_start=year_start,
                         year_end=year_end,
                         time_step=time_step,
                         file_path=__file__,
                         run_usecase=run_usecase,
                         execution_engine=execution_engine)

        self.test_post_procs = False

    def setup_usecase(self, study_folder_path=None):
        self.test_post_procs = False
        data_witness = super().setup_usecase()

        # update fossil invest & utilization ratio lower bound to not be too low
        min_invest = 1.
        max_invest = 8000.
        dspace_invests = {
            'fossil.FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix': [300., 300., 5000., True],
            f"{GlossaryCore.clean_energy}.{GlossaryCore.CleanEnergySimpleTechno}.{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_array_mix": [min_invest, min_invest, max_invest, True],
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix': [min_invest, min_invest, max_invest, False],
            'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix': [min_invest, min_invest, max_invest, False],
            'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix': [min_invest, min_invest, max_invest, False],
        }
        dspace_invests = self.make_dspace_invests(dspace_invests)
        min_UR = 50.
        dspace_UR = {
            'fossil_FossilSimpleTechno_utilization_ratio_array': [min_UR, min_UR, 100., True],
            f"{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_utilization_ratio_array": [min_UR, min_UR, 100., True],
            'carbon_capture.direct_air_capture.DirectAirCaptureTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],
            'carbon_capture.flue_gas_capture.FlueGasTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],
            'carbon_storage.CarbonStorageTechno_utilization_ratio_array': [min_UR, min_UR, 100., False],

        }
        dspace_UR = self.make_dspace_utilization_ratio(dspace_UR)

        # dspace pour Ine
        dspace_Ine = self.make_dspace_Ine()

        # start from last iteration found in the optim assumed to be the optimum (design_space_out should but does not provide the optimum)
        # from https://integration.osc-tsa.com/study-workspace?studyId=4526, tp=3.5°C
        # fossil
        dspace_invests.at[0, 'value'] = [839.0, 1289.7326346513792, 1419.8141521485918, 2306.3997405095997, 2523.299406293027, 1853.7557871578695, 927.6283550403235]
        dspace_UR.at[0, 'value'] = [100.0, 99.43578156885647, 85.41255175113776, 63.019677004996815, 70.55639618028486, 73.59652331964561, 78.9711387908367, 77.32474888789574, 70.17012518320456, 62.43011638108099]
        # renewalbe
        dspace_invests.at[1, 'value'] = [1259.0, 1906.9769903242482, 1958.200728325605, 1877.2621735761802, 1595.2683128462338, 944.6238770738995, 1.0]
        dspace_UR.at[1, 'value'] = [100.0, 65.84513878360015, 68.73521893993595, 66.82431187750439, 53.334856359388446, 50.83969255964565, 50.00512248187553, 50.0, 51.99755361635379, 50.545078745107936]
        # DAC UR
        dspace_invests.at[2, 'value'] = [1.3333, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06]
        dspace_UR.at[2, 'value'] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        # flue gas UR
        dspace_invests.at[3, 'value'] = [1.3333, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06]
        dspace_UR.at[3, 'value'] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        # CS UR
        dspace_invests.at[4, 'value'] = [1.3333, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06]
        dspace_UR.at[4, 'value'] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        # INE
        dspace_Ine.at[0, 'value'] = [25.5, 30.0, 30.0, 28.266188087516344, 26.92355374206309, 24.5, 25.887477728784216]

        '''
        # from https://integration.osc-tsa.com/study-workspace?studyId=4527 , tp=6°C
        dspace_UR.at[0, 'value'] = [100.0, 100.0, 95.51430461587898, 71.73800611249315, 71.76121810275187, 79.0548466213939, 83.75520557037733, 81.82783022742535, 81.41076250264636, 76.79493761714657]
        dspace_UR.at[1, 'value'] = [100.0, 76.48658289696345, 50.68673185345673, 50.0, 50.0, 57.018480266749535, 50.0, 50.03021016792428, 50.290632829019245, 51.64066610276135]
        dspace_Ine.at[0, 'value'] = [25.5, 30.0, 30.0, 30.0, 26.667183562922308, 24.795869808115732, 24.51802044939126]
        '''

        dspace = pd.concat([dspace_invests, dspace_UR, dspace_Ine], ignore_index=True)
        # update design var descriptor with Ine variable
        dvar_descriptor = data_witness[f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables.design_var_descriptor']
        design_var_descriptor_ine_variable = self.get_ine_dvar_descr()

        dvar_descriptor.update({
            "share_non_energy_invest_ctrl": design_var_descriptor_ine_variable
        })

        # Deactivate damage
        updated_data = {
            f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.assumptions_dict': {
                'compute_gdp': True,
                'compute_climate_impact_on_gdp': False,
                'activate_climate_effect_population': False,
                'activate_pandemic_effects': False
            },
            f'{self.study_name}.{self.optim_name}.design_space': dspace,
        }
        data_witness.update(updated_data)

        # Put low tax
        data_witness.update({
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.ccs_price_percentage": 0.0,
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.co2_damage_price_percentage": 0.0,
            f"{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.share_non_energy_invest_ctrl": np.array([27.0] * (GlossaryCore.NB_POLES_COARSE - 1)),
            f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.{self.witness_uc.extra_name}.Damage.tp_a3': 3.5,
            f'{self.study_name}.{self.optim_name}.{self.witness_uc.coupling_name}.tolerance': 1.e-12,
            f'{self.study_name}.{self.optim_name}.max_iter': 1,
        })

        return data_witness


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.test()
    '''
    from sostrades_core.tools.post_processing.post_processing_factory import (
        PostProcessingFactory)
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()
    ns = f'usecase_1_test_gradients_tp_3_5.WITNESS_MDO.WITNESS_Eval.WITNESS'
    filters = ppf.get_post_processing_filters_by_namespace(uc_cls.ee, ns)

    graph_list = ppf.get_post_processing_by_namespace(uc_cls.ee, ns, filters, as_json=False)
    for graph in graph_list:
        graph.to_plotly().show()
    '''
