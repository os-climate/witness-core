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
from climateeconomics.glossarycore import GlossaryCore
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sostrades_core.study_manager.study_manager import StudyManager

from os.path import join, dirname
from numpy import asarray, arange, array
import pandas as pd
import numpy as np
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def update_dspace_with(dspace_dict, name, value, lower, upper):
    ''' type(value) has to be ndarray
    '''
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)
    dspace_dict['variable'].append(name)
    dspace_dict['value'].append(value.tolist())
    dspace_dict['lower_bnd'].append(lower)
    dspace_dict['upper_bnd'].append(upper)
    dspace_dict['dspace_size'] += len(value)


def update_dspace_dict_with(dspace_dict, name, value, lower, upper, activated_elem=None, enable_variable=True):
    if not isinstance(lower, (list, np.ndarray)):
        lower = [lower] * len(value)
    if not isinstance(upper, (list, np.ndarray)):
        upper = [upper] * len(value)

    if activated_elem is None:
        activated_elem = [True] * len(value)
    dspace_dict[name] = {'value': value,
                         'lower_bnd': lower, 'upper_bnd': upper, 'enable_variable': enable_variable,
                         'activated_elem': activated_elem}

    dspace_dict['dspace_size'] += len(value)


class Study(StudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, name='', execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.study_name = 'usecase'
        self.agriculture_name = name
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.nb_poles = 8

    def setup_usecase(self):
        setup_data_list = []

        years = np.arange(self.year_start, self.year_end + 1, 1)
        year_range = self.year_end - self.year_start + 1

        # SSP1
        #         emissions_image_df = pd.DataFrame({
        #             'year': [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
        #             'emissions': [40.069004, 42.653234, 43.778496, 42.454758, 41.601928, 39.217532, 33.392294, 28.618414, 24.612914]})
        # SSP3
        #         emissions_image_df = pd.DataFrame({
        #             'year': [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
        #             'emissions': [48.502707, 55.000887, 59.877305, 64.001363, 67.962295, 71.792666, 75.571458, 80.137183, 85.214966]})
        # SSP5
        #         emissions_image_df = pd.DataFrame({
        #             'year': [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
        #             'emissions': [44.610389, 56.726452, 69.861617, 84.436466, 101.301616, 117.499826, 129.499348, 130.397532, 126.097683]})
        #
        #        f2 = interp1d(emissions_image_df['year'], emissions_image_df['emissions'])
        # Find values for 2020, 2050 and concat dfs
        #         emissions = f2(years)
        #         emissions_df = pd.DataFrame({GlossaryCore.Years: years, 'total_emissions':emissions, 'cum_total_emissions': np.zeros(year_range)})
        #         emissions_df.index = years
        #         # carbon emissions df

        # 460 bnT during 10 years
        #         cum_start = 460.0 / 44 * 12
        #
        #         emissions = list(np.ones(10) * cum_start / 10.0) + \
        #             list(np.array(np.linspace(0, 0, year_range - 10)))

        emissions = list(np.linspace(38.3, 0, 10)) + \
                    list(np.zeros(year_range - 10))
        emissions_df = pd.DataFrame({GlossaryCore.Years: years, 'total_emissions': emissions,
                                     'cum_total_emissions': np.zeros(year_range)})
        emissions_df.index = years
        # missing here the initial level of cumulated emissions
        emissions_df['cum_total_emissions'] = emissions_df['total_emissions'].cumsum()
        #         plt.plot(years, emissions_df['total_emissions'].values)
        #         plt.show()

        # private values economics operator model
        climate_input = {}
        climate_input[f"{self.study_name}.{GlossaryCore.YearStart}"] = self.year_start
        climate_input[f"{self.study_name}.{GlossaryCore.YearEnd}"] = self.year_end
        climate_input[f"{self.study_name}.{GlossaryCore.CO2EmissionsDfValue}"] = emissions_df

        setup_data_list.append(climate_input)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    # uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    # uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()

    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.sos_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(
            disc)
        graph_list = ppf.get_post_processing_by_discipline(
            disc, filters, as_json=False)

#         for graph in graph_list:
#             graph.to_plotly().show()
