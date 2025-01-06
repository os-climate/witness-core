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

import pandas as pd

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_7_witness_coarse_mda_gdp_model_w_damage_w_co2_tax import (
    Study as usecase7,
)


class Study(ClimateEconomicsStudyManager):
    USECASE7 = 'NZE'

    def __init__(self, bspline=False, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.bspline = bspline
        self.data_dir = join(dirname(__file__), 'data')
        self.check_outputs = False

    def setup_usecase(self, study_folder_path=None):

        self.driver_name = 'AnalysisWITNESS'

        scenario_dict = {self.USECASE7: usecase7(execution_engine=self.execution_engine)}

        '''
        NZE inspired: Net Zero Emissions just for the energy sector, ie CO2 emissions = 0 for the energy sector
            for the other sectors, it is non-zero
        NZE:  Net Zero Emissions for all sectors, therefore the energy sector captures and stores CO2 (therefore 
            the energy sector has a negative CO2 emission balance to compensate the non zero emissions of the other sectors
        '''

        input_selection = {
            "selected_input": [False, False, False, False, False, False, False, False, True],
            "full_name": [
                f"{self.USECASE7}.RenewableTechnoInfo.Opex_percentage",
                f"{self.USECASE7}.RenewableTechnoInfo.Initial_capex",
                f"{self.USECASE7}.RenewableTechnoInfo.Energy_costs",
                f"{self.USECASE7}.FossilTechnoInfo.Opex_percentage",
                f"{self.USECASE7}.FossilTechnoInfo.Initial_capex",
                f"{self.USECASE7}.FossilTechnoInfo.Energy_costs",
                f"{self.USECASE7}.FossilTechnoInfo.CO2_from_production",
                f"{self.USECASE7}.Damage.tp_a3",
                f"{self.USECASE7}.Temperature change.init_temp_atmo",
                          ],
        }
        input_selection = pd.DataFrame(input_selection)

        output_selection = {
            "selected_output": [True, True, True, True, True, True, True, True, True],
            "full_name": [
                f"{self.USECASE7}.Indicators.mean_energy_price_2100",
                f"{self.USECASE7}.Indicators.fossil_energy_price_2100",
                f"{self.USECASE7}.Indicators.renewable_energy_price_2100",
                f"{self.USECASE7}.Indicators.total_energy_production_2100",
                f"{self.USECASE7}.Indicators.fossil_energy_production_2100",
                f"{self.USECASE7}.Indicators.renewable_energy_production_2100",
                f"{self.USECASE7}.Indicators.world_net_product_2100",
                f"{self.USECASE7}.Indicators.temperature_rise_2100",
                f"{self.USECASE7}.Indicators.welfare_indicator",
                          ],
        }
        output_selection = pd.DataFrame(output_selection)

        values_dict = {}
        values_dict[f'{self.study_name}.SampleGenerator.sampling_method'] = "tornado_chart_analysis"
        values_dict[f'{self.study_name}.SampleGenerator.variation_list'] = [-5., 5.]
        values_dict[f'{self.study_name}.{self.driver_name}.with_sample_generator'] = True
        values_dict[f'{self.study_name}.SampleGenerator.eval_inputs'] = input_selection
        values_dict[f'{self.study_name}.{self.driver_name}.gather_outputs'] = output_selection


        # setup each scenario (mda settings ignored)
        for scenario, uc in scenario_dict.items():
            uc.study_name = f'{self.study_name}.{self.driver_name}.{scenario}'
            for dict_data in uc.setup_usecase():
                values_dict.update(dict_data)
            # NB: switch to MDAGaussSeidel so it won't crash with the l1 tests, GSNewton diverges.
            values_dict[f'{self.study_name}.{self.driver_name}.{scenario}.inner_mda_name'] = 'MDANewtonRaphson'
            values_dict[f'{self.study_name}.{self.driver_name}.{scenario}.max_mda_iter'] = 2
        return values_dict

if '__main__' == __name__:
    uc_cls = Study()
    # uc_cls.load_data()
    # uc_cls.run()
    uc_cls.test()
    # post_processing_factory = PostProcessingFactory()
    # post_processing_factory.get_post_processing_by_namespace(
    #     uc_cls.execution_engine, f'{uc_cls.study_name}.Post-processing', [])
    # all_post_processings = post_processing_factory.get_all_post_processings(
    #      uc_cls.execution_engine, False, as_json=False, for_test=False)

#    for namespace, post_proc_list in all_post_processings.items():
#        for chart in post_proc_list:
#            chart.to_plotly().show()
