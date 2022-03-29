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
from energy_models.core.energy_mix_study_manager import EnergyMixStudyManager
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS,\
    INVEST_DISCIPLINE_DEFAULT

DEFAULT_TECHNOLOGIES_LIST = ['Crop', 'Forest']
TECHNOLOGIES_LIST_FOR_OPT = ['Crop', 'Forest']
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


class Study(EnergyMixStudyManager):
    def __init__(self, year_start=2020, year_end=2100, time_step=1, technologies_list=TECHNOLOGIES_LIST_FOR_OPT,
                 bspline=True,  main_study=True, execution_engine=None, invest_discipline=INVEST_DISCIPLINE_DEFAULT):
        super().__init__(__file__, technologies_list=technologies_list,
                         main_study=main_study, execution_engine=execution_engine, invest_discipline=invest_discipline)
        self.year_start = year_start
        self.year_end = year_end
        self.years = np.arange(self.year_start, self.year_end + 1)
        self.energy_name = None
        self.bspline = bspline

    def get_investments(self):
        invest_biomass_dry_mix_dict = {}
        l_ctrl = np.arange(0, 8)

        if 'Forest' in self.technologies_list:
            invest_biomass_dry_mix_dict['Forest'] = [
                (1 + 0.03)**i for i in l_ctrl]

        if 'Crop' in self.technologies_list:
            invest_biomass_dry_mix_dict['Crop'] = np.array([
                1.0, 1.0, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4])

        if self.bspline:
            invest_biomass_dry_mix_dict['years'] = self.years

            for techno in self.technologies_list:
                invest_biomass_dry_mix_dict[techno], _ = self.invest_bspline(
                    invest_biomass_dry_mix_dict[techno], len(self.years))

        biomass_dry_mix_invest_df = pd.DataFrame(invest_biomass_dry_mix_dict)

        return biomass_dry_mix_invest_df

    def setup_usecase(self):
        agriculture_mix = 'AgricultureMix'
        energy_name = f'{agriculture_mix}'
        years = np.arange(self.year_start, self.year_end + 1)
        # reference_data_name = 'Reference_aircraft_data'
        self.energy_prices = pd.DataFrame({'years': years,
                                           'electricity': 16.0})
        year_range = self.year_end - self.year_start + 1
        population = np.array(
            [7886.69358, 7966.665211, 8045.375451, 8122.797867, 8198.756532, 8273.083818, 8345.689982, 8416.57613,
             8485.795919, 8553.312856, 8619.058953, 8683.042395, 8745.257656,
             8805.680119, 8864.337481, 8921.246891, 8976.395584, 9029.771873, 9081.354412, 9131.121464, 9179.083525,
             9225.208209, 9269.477788, 9311.832053, 9352.201944, 9390.558602,
             9426.867911, 9461.124288, 9493.330078, 9523.465887, 9551.506077, 9577.443894, 9601.291404, 9623.075287,
             9642.80847, 9660.498856, 9676.158366, 9689.826469, 9701.54988,
             9711.366041, 9719.318272, 9725.419042, 9729.702777, 9732.206794, 9732.922689, 9731.871402, 9729.064465,
             9724.513081, 9718.249401, 9710.282554, 9700.610324, 9689.251038,
             9676.243561, 9661.590658, 9645.329918, 9627.498797, 9608.104964, 9587.197508, 9564.828118, 9541.038722,
             9515.888869, 9489.415825, 9461.693469, 9432.803085, 9402.775341,
             9371.660258, 9339.478398, 9306.261187, 9272.043294, 9236.831993, 9200.632391, 9163.429244, 9125.227987,
             9086.036337, 9045.87235, 9004.753844, 8962.700979, 8919.730768,
             8875.855926, 8831.098444, 8785.553666])
        temperature = np.array(np.linspace(1.05, 5, year_range))
        #         temperature = np.array(np.linspace(1.05, 1.05, year_range))

        temperature_df = pd.DataFrame(
            {"years": years, "temp_atmo": temperature})
        temperature_df.index = years

        population_df = pd.DataFrame(
            {"years": years, "population": population})
        population_df.index = years

        red_to_white_meat = np.linspace(0, 50, year_range)
        meat_to_vegetables = np.linspace(0, 50, year_range)
        red_to_white_meat_df = pd.DataFrame(
            {'years': years, 'red_to_white_meat_percentage': red_to_white_meat})
        meat_to_vegetables_df = pd.DataFrame(
            {'years': years, 'meat_to_vegetables_percentage': meat_to_vegetables})
        red_to_white_meat_df.index = years
        meat_to_vegetables_df.index = years
        self.red_to_white_meat_df = red_to_white_meat_df
        self.meat_to_vegetables_df = meat_to_vegetables_df

        diet_df = pd.DataFrame({'red meat': [11.02],
                                'white meat': [31.11],
                                'milk': [79.27],
                                'eggs': [9.68],
                                'rice and maize': [97.76],
                                'potatoes': [32.93],
                                'fruits and vegetables': [217.62],
                                })
        other = np.array(np.linspace(0.102, 0.102, year_range))

        # the value for invest_level is just set as an order of magnitude
        self.invest_level = pd.DataFrame(
            {'years': years, 'invest': 1e4})

        land_surface_for_food = pd.DataFrame({'years': years,
                                              'Agriculture total (Gha)': np.ones(len(years)) * 4.8})

        self.margin = pd.DataFrame(
            {'years': years, 'margin': np.ones(len(years)) * 110.0})
        # From future of hydrogen
        self.transport = pd.DataFrame(
            {'years': years, 'transport': np.ones(len(years)) * 7.6})

        self.resources_price = pd.DataFrame(columns=['years', 'CO2', 'water'])
        self.resources_price['years'] = years
        self.resources_price['CO2'] = np.linspace(
            50.0, 100.0, len(years))         # biomass_dry price in $/kg
        self.energy_carbon_emissions = pd.DataFrame(
            {'years': years, 'biomass_dry': - 0.64 / 4.86, 'solid_fuel': 0.64 / 4.86, 'electricity': 0.0, 'methane': 0.123 / 15.4, 'syngas': 0.0, 'hydrogen.gaseous_hydrogen': 0.0, 'crude oil': 0.02533})

        deforestation_surface = np.linspace(10, 5, year_range)
        self.deforestation_surface_df = pd.DataFrame(
            {"years": years, "deforested_surface": deforestation_surface})

        forest_invest = np.linspace(5, 8, year_range)
        self.forest_invest_df = pd.DataFrame(
            {"years": years, "forest_investment": forest_invest})

        # define invest mix
        investment_mix = self.get_investments()

        values_dict = {f'{self.study_name}.year_start': self.year_start,
                       f'{self.study_name}.year_end': self.year_end,
                       f'{self.study_name}.{energy_name}.technologies_list': self.technologies_list,
                       f'{self.study_name}.margin': self.margin,
                       f'{self.study_name}.transport_cost': self.transport,
                       f'{self.study_name}.transport_margin': self.margin,
                       f'{self.study_name}.invest_techno_mix': investment_mix,
                       }
        if self.main_study:
            values_dict.update(
                {f'{self.study_name}.{energy_name}.Crop.land_surface_for_food_df': land_surface_for_food,
                 f'{self.study_name}.{energy_name}.Crop.diet_df': diet_df,
                 f'{self.study_name}.{energy_name}.Crop.red_to_white_meat': red_to_white_meat,
                 f'{self.study_name}.{energy_name}.Crop.meat_to_vegetables': meat_to_vegetables,
                 f'{self.study_name}.{energy_name}.Crop.other_use_crop': other,
                 f'{self.study_name}.deforestation_surface': self.deforestation_surface_df,
                 f'{self.study_name}.forest_investment': self.forest_invest_df,
                 f'{self.study_name}.managed_wood_investment': self.forest_invest_df,
                 f'{self.study_name}.unmanaged_wood_investment': self.forest_invest_df,
                 f'{self.study_name}.population_df': population_df,
                 f'{self.study_name}.temperature_df': temperature_df,
                 })


            if self.invest_discipline == INVEST_DISCIPLINE_OPTIONS[1]:
                investment_mix_sum = investment_mix.drop(
                    columns=['years']).sum(axis=1)
                for techno in self.technologies_list:
                    invest_level_techno = pd.DataFrame({'years': self.invest_level['years'].values,
                                                        'invest': self.invest_level['invest'].values * investment_mix[techno].values / investment_mix_sum})
                    values_dict[f'{self.study_name}.{energy_name}.{techno}.invest_level'] = invest_level_techno
            else:
                values_dict[f'{self.study_name}.invest_level'] = self.invest_level
        else:
            self.update_dv_arrays()


        return [values_dict]



    def setup_design_space(self):
            #-- energy optimization inputs
            # Design Space
        dim_a = len(
            self.red_to_white_meat_df['red_to_white_meat_percentage'].values)
        lbnd1 = [0.0] * dim_a
        ubnd1 = [70.0] * dim_a

        # Design variables:
        self.update_dspace_dict_with(
            'red_to_white_meat_array', self.red_to_white_meat_df['red_to_white_meat_percentage'].values, lbnd1, ubnd1)
        self.update_dspace_dict_with(
            'meat_to_vegetables_array', self.meat_to_vegetables_df['meat_to_vegetables_percentage'].values, lbnd1, ubnd1)

    def setup_design_space_ctrl_new(self):
        # Design Space
        #header = ['variable', 'value', 'lower_bnd', 'upper_bnd']
        ddict = {}
        ddict['dspace_size'] = 0

        # Design variables:
        update_dspace_dict_with(ddict, 'red_to_white_meat_ctrl',
                                list(self.design_space_ctrl['red_to_white_meat_ctrl'].values), [0.0] * self.nb_poles, [70.0] * self.nb_poles, activated_elem=[True, True, True, True, True, True, True])
        update_dspace_dict_with(ddict, 'meat_to_vegetables_ctrl',
                                list(self.design_space_ctrl['meat_to_vegetables_ctrl'].values), [0.0] * self.nb_poles, [70.0] * self.nb_poles, activated_elem=[True, True, True, True, True, True, True])

        return ddict

if '__main__' == __name__:
    uc_cls = Study(main_study=True,
                   technologies_list=DEFAULT_TECHNOLOGIES_LIST)
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
