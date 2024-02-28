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
from os.path import join, dirname

import pandas as pd

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_2_witness_coarse_mda_gdp_model_wo_damage_wo_co2_tax import \
    Study as usecase2
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_2b_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import \
    Study as usecase2b
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_3_witness_coarse_mda_gdp_model_wo_damage_w_co2_tax import \
    Study as usecase3
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_4_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import \
    Study as usecase4
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_5_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import \
    Study as usecase5
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_6_witness_coarse_mda_gdp_model_w_damage_wo_co2_tax import \
    Study as usecase6
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_7_witness_coarse_mda_gdp_model_w_damage_w_co2_tax import \
    Study as usecase7
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev.usecase_witness_coarse_new import \
    Study as usecase_witness_mda
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from climateeconomics.glossarycore import GlossaryCore


class Study(ClimateEconomicsStudyManager):

    USECASE2 = '- damage - tax, fossil 100%'
    USECASE2B ='+ damage - tax, fossil 100%'
    USECASE3 = '- damage + tax, IEA'
    USECASE4 = '+ damage - tax, fossil 40%'
    USECASE5 = '+ damage - tax, STEP inspired'
    USECASE6 = '+ damage - tax, NZE inspired'
    USECASE7 = '+ damage + tax, NZE'

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.check_outputs = True

    def setup_usecase(self, study_folder_path=None):

        self.scatter_scenario = 'mda_scenarios'

        scenario_dict = {self.USECASE2: usecase2(execution_engine=self.execution_engine),
                         self.USECASE2B: usecase2b(execution_engine=self.execution_engine),
                         self.USECASE3: usecase3(execution_engine=self.execution_engine),
                         self.USECASE4: usecase4(execution_engine=self.execution_engine),
                         self.USECASE5: usecase5(execution_engine=self.execution_engine),
                         self.USECASE6: usecase6(execution_engine=self.execution_engine),
                         self.USECASE7: usecase7(execution_engine=self.execution_engine),
                         }

        '''
        NZE inspired: Net Zero Emissions just for the energy sector, ie CO2 emissions = 0 for the energy sector
            for the other sectors, it is non-zero
        NZE:  Net Zero Emissions for all sectors, therefore the energy sector captures and stores CO2 (therefore 
            the energy sector has a negative CO2 emission balance to compensate the non zero emissions of the other sectors
        '''

        scenario_list = list(scenario_dict.keys())
        values_dict = {}

        scenario_df = pd.DataFrame({'selected_scenario': [True] * len(scenario_list),
                                    'scenario_name': scenario_list})
        values_dict[f'{self.study_name}.{self.scatter_scenario}.samples_df'] = scenario_df
        values_dict[f'{self.study_name}.{self.scatter_scenario}.scenario_list'] = scenario_list

        # setup mda
        uc_mda = usecase_witness_mda(execution_engine=self.execution_engine)
        uc_mda.study_name = self.study_name  # mda settings on root coupling
        values_dict.update(uc_mda.setup_mda())
        # assumes max of 16 cores per computational node
        values_dict[f'{self.study_name}.n_subcouplings_parallel'] = min(16, len(scenario_df.loc[scenario_df['selected_scenario']==True]))
        # setup each scenario (mda settings ignored)
        for scenario, uc in scenario_dict.items():
            uc.study_name = f'{self.study_name}.{self.scatter_scenario}.{scenario}'
            for dict_data in uc.setup_usecase():
                values_dict.update(dict_data)
        return values_dict

    def specific_check_outputs(self):
        """Some outputs are retrieved and their range is checked"""
        list_scenario = {self.USECASE2, self.USECASE2B, self.USECASE3, self.USECASE4, self.USECASE5, self.USECASE6, self.USECASE7}
        dm = self.execution_engine.dm
        all_temp_increase = dm.get_all_namespaces_from_var_name('temperature_df')
        ref_value_temp_increase= {self.USECASE2: 4.23, self.USECASE2B: 4.07, self.USECASE3: 3.14, self.USECASE4: 3.34, self.USECASE5: 2.86, self.USECASE6: 2.58, self.USECASE7: 2.41}
        all_co2_taxes = dm.get_all_namespaces_from_var_name('CO2_taxes')
        ref_value_co2_tax = {self.USECASE2: 0, self.USECASE2B: 0, self.USECASE3: 344, self.USECASE4: 0, self.USECASE5: 0, self.USECASE6: 0, self.USECASE7: 1192}
        for scenario in list_scenario:
            for scenario_temp_increase in all_temp_increase:
                if scenario in scenario_temp_increase:
                    temp_increase = dm.get_value(scenario_temp_increase)
                    value_temp_increase = temp_increase.loc[temp_increase['years']==2100]['temp_atmo'].values[0]
                    print("Temperature increase by 2100 for scenario ", scenario, ":", value_temp_increase, " degrees")
                    assert value_temp_increase >= ref_value_temp_increase[scenario] * 0.8
                    assert value_temp_increase <= ref_value_temp_increase[scenario] * 1.2
            for scenario_co2_tax in all_co2_taxes:
                if scenario in scenario_co2_tax:
                    co2_tax = dm.get_value(scenario_co2_tax)
                    value_co2_tax = co2_tax.loc[co2_tax['years']==2100]['CO2_tax'].values[0]
                    print("CO2_tax in 2100 for scenario ", scenario, ":", value_co2_tax)
                    assert value_co2_tax >= ref_value_co2_tax[scenario] * 0.8
                    assert value_co2_tax <= ref_value_co2_tax[scenario] * 1.2


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
    # post_processing_factory = PostProcessingFactory()
    # post_processing_factory.get_post_processing_by_namespace(
    #     uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing', [])
    # all_post_processings = post_processing_factory.get_all_post_processings(
    #      uc_cls.execution_engine, False, as_json=False, for_test=False)

#    for namespace, post_proc_list in all_post_processings.items():
#        for chart in post_proc_list:
#            chart.to_plotly().show()
