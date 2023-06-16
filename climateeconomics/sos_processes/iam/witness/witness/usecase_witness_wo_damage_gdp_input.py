'''
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


from climateeconomics.sos_processes.iam.witness.witness.usecase_witness import Study as usecase_witness

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager


from pandas import DataFrame
from numpy import arange, linspace


class Study(ClimateEconomicsStudyManager):

    def __init__(self, run_usecase = False, execution_engine=None, year_start = 2020, year_end = 2100, time_step = 1):
        super().__init__(__file__,  run_usecase = run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end 
        self.time_step = time_step

    def setup_usecase(self):
        

        witness_uc = usecase_witness()
        witness_uc.study_name = self.study_name
        data_witness = witness_uc.setup_usecase()
        years = arange(self.year_start, self.year_end + 1, self.time_step)
        gross_output = linspace(85,145, len(years))
        df_gross_output = DataFrame({'years':years, 
                                     'gross_output': gross_output})
        updated_data = {f'{self.study_name}.assumptions_dict': {'compute_gdp': False,
                                                                'compute_climate_impact_on_gdp': False,
                                                                'activate_climate_effect_population': True,
                                                                'invest_co2_tax_in_renewables': False
                                                                },
                        f'{self.study_name}.gross_output_in': df_gross_output}
        data_witness.append(updated_data)
        return data_witness


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
