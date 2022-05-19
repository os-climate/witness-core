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
import numpy as np
import pandas as pd
import scipy.interpolate as sc
from numpy import asarray, arange, array

from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from sos_trades_core.study_manager.study_manager import StudyManager
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from climateeconomics.sos_processes.iam.witness.forest_v2_process.usecase import Study as datacase_forest

AGRI_MIX_MODEL_LIST = ['Crop', 'Forest']
AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT = [
    'ManagedWood', 'UnmanagedWood', 'CropEnergy']
COARSE_AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT = []


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
                         'lower_bnd': lower, 'upper_bnd': upper, 'enable_variable': enable_variable, 'activated_elem': activated_elem}

    dspace_dict['dspace_size'] += len(value)


class Study(StudyManager):
    def __init__(self, year_start=2020, year_end=2100, time_step=1, execution_engine=None, agri_techno_list=AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
                 model_list=AGRI_MIX_MODEL_LIST):
        super().__init__(__file__, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.techno_list = agri_techno_list
        self.model_list = model_list
        self.energy_name = None
        self.nb_poles = 8
        self.additional_ns = ''

    def setup_usecase(self):

        agriculture_mix = 'AgricultureMix'
        energy_name = f'{agriculture_mix}'
        years = np.arange(self.year_start, self.year_end + 1)
        self.energy_prices = pd.DataFrame({'years': years,
                                           'electricity': 16.0})
        year_range = self.year_end - self.year_start + 1

        temperature = np.array(np.linspace(1.05, 5.0, year_range))
        temperature_df = pd.DataFrame(
            {"years": years, "temp_atmo": temperature})
        temperature_df.index = years

        population = np.array(np.linspace(7800.0, 9000.0, year_range))
        population_df = pd.DataFrame(
            {"years": years, "population": population})
        population_df.index = years

        red_meat_percentage = np.linspace(6.82, 1, year_range)
        white_meat_percentage = np.linspace(13.95, 5, year_range)
        self.red_meat_percentage = pd.DataFrame({
            'years': years,
            'red_meat_percentage': red_meat_percentage})
        self.white_meat_percentage = pd.DataFrame({
            'years': years,
            'white_meat_percentage': white_meat_percentage})

        diet_df = pd.DataFrame({'red meat': [11.02],
                                'white meat': [31.11],
                                'milk': [79.27],
                                'eggs': [9.68],
                                'rice and maize': [97.76],
                                'potatoes': [32.93],
                                'fruits and vegetables': [217.62],
                                })
        other = np.array(np.linspace(0.102, 0.102, year_range))

        self.margin = pd.DataFrame(
            {'years': years, 'margin': np.ones(len(years)) * 110.0})
        # From future of hydrogen
        self.transport = pd.DataFrame(
            {'years': years, 'transport': np.ones(len(years)) * 7.6})

        self.energy_carbon_emissions = pd.DataFrame(
            {'years': years, 'biomass_dry': - 0.64 / 4.86, 'solid_fuel': 0.64 / 4.86, 'electricity': 0.0, 'methane': 0.123 / 15.4, 'syngas': 0.0, 'hydrogen.gaseous_hydrogen': 0.0, 'crude oil': 0.02533})

        deforestation_surface = np.linspace(10, 5, year_range)
        self.deforestation_surface_df = pd.DataFrame(
            {"years": years, "deforested_surface": deforestation_surface})

        forest_invest = np.linspace(5, 8, year_range)

        self.forest_invest_df = pd.DataFrame(
            {"years": years, "forest_investment": forest_invest})

        if 'CropEnergy' in self.techno_list:
            crop_invest = np.linspace(0.5, 0.25, year_range)
        else:
            crop_invest = [0] * year_range
        if 'ManagedWood' in self.techno_list:
            mw_invest = np.linspace(1, 4, year_range)
        else:
            mw_invest = [0] * year_range
        if 'UnmanagedWood' in self.techno_list:
            uw_invest = np.linspace(0, 1, year_range)
        else:
            uw_invest = [0] * year_range
        self.uw_invest_df = pd.DataFrame(
            {"years": years, "investment": uw_invest})
        self.mw_invest_df = pd.DataFrame(
            {"years": years, "investment": mw_invest})
        self.crop_investment = pd.DataFrame(
            {'years': years, 'investment': crop_invest})
        deforest_invest = np.linspace(10, 1, year_range)
        deforest_invest_df = pd.DataFrame(
            {"years": years, "investment": deforest_invest})

        co2_taxes_year = [2018, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
        co2_taxes = [14.86, 17.22, 20.27,
                     29.01,  34.05,   39.08,  44.69,   50.29]
        func = sc.interp1d(co2_taxes_year, co2_taxes,
                           kind='linear', fill_value='extrapolate')

        self.co2_taxes = pd.DataFrame(
            {'years': years, 'CO2_tax': func(years)})

        values_dict = {
            f'{self.study_name}.year_start': self.year_start,
            f'{self.study_name}.year_end': self.year_end,
            f'{self.study_name}.{energy_name}.technologies_list': self.model_list,
            f'{self.study_name}.margin': self.margin,
            f'{self.study_name}.transport_cost': self.transport,
            f'{self.study_name}.transport_margin': self.margin,
            f'{self.study_name}.CO2_taxes': self.co2_taxes,
            f'{self.study_name}.{energy_name}.Crop.diet_df': diet_df,
            f'{self.study_name}.{energy_name}.Crop.red_meat_percentage': self.red_meat_percentage,
            f'{self.study_name}.{energy_name}.Crop.white_meat_percentage': self.white_meat_percentage,
            f'{self.study_name}.{energy_name}.Crop.other_use_crop': other,
            f'{self.study_name}.{energy_name}.Crop.crop_investment': self.crop_investment,
            f'{self.study_name}.deforestation_surface': self.deforestation_surface_df,
            f'{self.study_name + self.additional_ns}.forest_investment': self.forest_invest_df,
            f'{self.study_name}.{energy_name}.Forest.managed_wood_investment': self.mw_invest_df,
            f'{self.study_name}.{energy_name}.Forest.deforestation_investment': deforest_invest_df,
            f'{self.study_name}.population_df': population_df,
            f'{self.study_name}.temperature_df': temperature_df
        }

        red_meat_percentage_ctrl = np.linspace(6.82, 6.82, self.nb_poles)
        white_meat_percentage_ctrl = np.linspace(13.95, 13.95, self.nb_poles)
        deforestation_investment_ctrl = np.linspace(10.0, 5.0, self.nb_poles)
        forest_investment_array_mix = np.linspace(5.0, 8.0, self.nb_poles)
        crop_investment_array_mix = np.linspace(1.0, 1.5, self.nb_poles)
        managed_wood_investment_array_mix = np.linspace(
            2.0, 3.0, self.nb_poles)
        unmanaged_wood_investment_array_mix = np.linspace(
            4.0, 5.0, self.nb_poles)

        design_space_ctrl_dict = {}
        design_space_ctrl_dict['red_meat_percentage_ctrl'] = red_meat_percentage_ctrl
        design_space_ctrl_dict['white_meat_percentage_ctrl'] = white_meat_percentage_ctrl
        design_space_ctrl_dict['deforestation_investment_ctrl'] = deforestation_investment_ctrl
        design_space_ctrl_dict['forest_investment_array_mix'] = forest_investment_array_mix

        if 'CropEnergy' in self.techno_list:
            design_space_ctrl_dict['crop_investment_array_mix'] = crop_investment_array_mix
        if 'ManagedWood' in self.techno_list:
            design_space_ctrl_dict['managed_wood_investment_array_mix'] = managed_wood_investment_array_mix
        if 'UnmanagedWood' in self.techno_list:
            design_space_ctrl_dict['unmanaged_wood_investment_array_mix'] = unmanaged_wood_investment_array_mix

        design_space_ctrl = pd.DataFrame(design_space_ctrl_dict)
        self.design_space_ctrl = design_space_ctrl
        self.dspace = self.setup_design_space_ctrl_new()

        return ([values_dict])

    def setup_design_space_ctrl_new(self):
        # Design Space
        # header = ['variable', 'value', 'lower_bnd', 'upper_bnd']
        ddict = {}
        ddict['dspace_size'] = 0

        # Design variables
        # -----------------------------------------
        # Crop related
        update_dspace_dict_with(ddict, 'red_meat_percentage_ctrl',
                                list(self.design_space_ctrl['red_meat_percentage_ctrl'].values), [1.0] * self.nb_poles, [10.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        update_dspace_dict_with(ddict, 'white_meat_percentage_ctrl',
                                list(self.design_space_ctrl['white_meat_percentage_ctrl'].values), [5.0] * self.nb_poles, [20.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)

        update_dspace_dict_with(ddict, 'deforestation_investment_ctrl',
                                list(self.design_space_ctrl['deforestation_investment_ctrl'].values), [0.0] * self.nb_poles, [100.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        # -----------------------------------------
        # Invests
        update_dspace_dict_with(ddict, 'forest_investment_array_mix',
                                list(self.design_space_ctrl['forest_investment_array_mix'].values), [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles, activated_elem=[True] * self.nb_poles)
        if 'CropEnergy' in self.techno_list:
            update_dspace_dict_with(ddict, 'crop_investment_array_mix',
                                    list(self.design_space_ctrl['crop_investment_array_mix'].values), [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles, activated_elem=[True] * self.nb_poles, enable_variable=False,)
        if 'ManagedWood' in self.techno_list:
            update_dspace_dict_with(ddict, 'managed_wood_investment_array_mix',
                                    list(self.design_space_ctrl['managed_wood_investment_array_mix'].values), [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles, activated_elem=[True] * self.nb_poles, enable_variable=False)
        if 'UnmanagedWood' in self.techno_list:
            update_dspace_dict_with(ddict, 'unmanaged_wood_investment_array_mix',
                                    list(self.design_space_ctrl['unmanaged_wood_investment_array_mix'].values), [1.0e-6] * self.nb_poles, [3000.0] * self.nb_poles, activated_elem=[True] * self.nb_poles, enable_variable=False,)

        return ddict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.sos_disciplines:
        filters = ppf.get_post_processing_filters_by_discipline(
            disc)
        graph_list = ppf.get_post_processing_by_discipline(
            disc, filters, as_json=False)

        # for graph in graph_list:
        #     graph.to_plotly().show()
