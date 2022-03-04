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
from copy import deepcopy


class Forest():
    """
    Forest model class 
    basic for now, to evolve 

    """
    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    TIME_STEP = 'time_step'
    LIMIT_DEFORESTATION_SURFACE = 'limit_deforestation_surface'
    DEFORESTATION_SURFACE = 'deforestation_surface'
    CO2_PER_HA = 'CO2_per_ha'
    INITIAL_CO2_EMISSIONS = 'initial_emissions'
    REFORESTATION_INVESTMENT = 'forest_investment'
    REFORESTATION_COST_PER_HA = 'reforestation_cost_per_ha'

    FOREST_SURFACE_DF = 'forest_surface_df'
    FOREST_DETAIL_SURFACE_DF = 'forest_surface_detail_df'
    CO2_EMITTED_FOREST_DF = 'CO2_emitted_forest_df'
    CO2_EMITTED_DETAIL_DF = 'CO2_emissions_detail_df'

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.set_data()
        self.create_dataframe()

    def set_data(self):
        self.year_start = self.param[self.YEAR_START]
        self.year_end = self.param[self.YEAR_END]
        self.time_step = self.param[self.TIME_STEP]
        # deforestation limite
        self.limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]
        # percentage of deforestation
        self.deforestation_surface = self.param[self.DEFORESTATION_SURFACE]
        # kg of CO2 not absorbed for 1 ha of forest deforested
        self.CO2_per_ha = self.param[self.CO2_PER_HA]
        # initial CO2 emissions
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        # forest data
        self.forest_investment = self.param[self.REFORESTATION_INVESTMENT]
        self.cost_per_ha = self.param[self.REFORESTATION_COST_PER_HA]
        self.techno_wood_info = self.param['wood_techno_dict']
        self.managed_wood_inital_prod = self.param['managed_wood_initial_prod']
        self.managed_wood_initial_surface = self.param['managed_wood_initial_surface']
        self.managed_wood_invest_before_year_start = self.param[
            'managed_wood_invest_before_year_start']
        self.managed_wood_investment = self.param['managed_wood_investment']
        self.unmanaged_wood_inital_prod = self.param['unmanaged_wood_initial_prod']
        self.unmanaged_wood_initial_surface = self.param['unmanaged_wood_initial_surface']
        self.unmanaged_wood_invest_before_year_start = self.param[
            'unmanaged_wood_invest_before_year_start']
        self.unmanaged_wood_investment = self.param['unmanaged_wood_investment']

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.forest_surface_df = pd.DataFrame()
        self.CO2_emitted_df = pd.DataFrame()
        self.managed_wood_df = pd.DataFrame()
        self.biomass_dry = pd.DataFrame()

    def compute(self, in_dict):
        """
        Computation methods
        """
        self.deforestation_surface = in_dict[self.DEFORESTATION_SURFACE]
        self.year_start = in_dict[self.YEAR_START]
        self.year_end = in_dict[self.YEAR_END]
        self.time_step = in_dict[self.TIME_STEP]
        self.forest_investment = in_dict[self.REFORESTATION_INVESTMENT]
        self.cost_per_ha = in_dict[self.REFORESTATION_COST_PER_HA]
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        self.limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]
        self.years = np.arange(
            self.year_start, self.year_end + 1, self.time_step)

        self.forest_surface_df['years'] = self.years
        self.managed_wood_df['years'] = self.years
        self.biomass_dry['years'] = self.years
        self.CO2_emitted_df['years'] = self.years

        # compute data of each contribution
        self.compute_reforestation_deforestation()
        self.compute_managed_wood_production()
        self.compute_reforestation_deforestation()
        # sum up global surface data
        self.sumup_global_surface_data()
        # check deforestation limit
        self.check_deforestation_limit()

        # sum up global CO2 data
        self.compute_global_CO2_production

    def compute_managed_wood_production(self):
        """
        compute data concerning managed wood : surface taken, production, CO2 absorbed, as delta and cumulative
        """
        construction_delay = self.techno_wood_info['construction_delay']
        density_per_ha = self.techno_wood_info['density_per_ha']
        mean_density = self.techno_wood_info['mean_density']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        residue_density_percentage = self.techno_wood_info['residue_density_percentage']
        residue_percentage_for_energy = self.techno_wood_info['residue_percentage_for_energy']
        # ADD TEST FOR $  or € unit OF PRICE #############
        mw_cost = self.techno_wood_info['Managed_wood_price_per_ha']
        # managed wood from past invest. invest in G$ - surface in Gha.
        mw_from_past_invest = self.managed_wood_invest_before_year_start['investment'] / mw_cost
        # managed wood from actual invest
        mw_from_invest = self.managed_wood_investment['investment'] / mw_cost
        # concat all managed wood form invest
        mw_added = pd.concat([mw_from_past_invest, mw_from_invest])
        # remove value that exceed year_end
        for i in range(0, construction_delay):
            mw_added = np.delete(mw_added, len(mw_added) - 1)

        # Surface part
        self.managed_wood_df['delta_surface'] = mw_added
        self.managed_wood_df['cumulative_surface'] = np.cumsum(
            mw_added) + self.managed_wood_initial_surface

        # Biomass production part
        ##### precise what is 3.6 ################
        self.managed_wood_df['delta_biomass_production'] = self.managed_wood_df['delta_surface'] * density_per_ha * mean_density * 3.6 / \
            years_between_harvest / (1 - recycle_part)
        self.managed_wood_df['biomass_production'] = np.cumsum(
            self.managed_wood_df['delta_biomass_production']) + self.managed_wood_inital_prod
        self.managed_wood_df['residues_production'] = self.managed_wood_df['biomass_production'] * \
            residue_density_percentage * (1 - residue_percentage_for_energy)

        # CO2 part
        self.managed_wood_df['delta_CO2_emitted'] = - \
            self.managed_wood_df['delta_surface'] * self.CO2_per_ha
        self.managed_wood_df['CO2_emitted'] = - \
            self.managed_wood_df['cumulative_surface'] * self.CO2_per_ha

    def compute_unmanaged_wood_production(self):
        """
        compute data concerning unmanaged wood : surface taken, production, CO2 absorbed, as delta and cumulative
        """

        construction_delay = self.techno_wood_info['construction_delay']
        density_per_ha = self.techno_wood_info['density_per_ha']
        mean_density = self.techno_wood_info['mean_density']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        residue_density_percentage = self.techno_wood_info['residue_density_percentage']
        residue_percentage_for_energy = self.techno_wood_info['residue_percentage_for_energy']
        # ADD TEST FOR $  or € unit OF PRICE #############
        uw_cost = self.techno_wood_info['Unmanaged_wood_price_per_ha']
        # unmanaged wood from past invest. invest in G$ - surface in Gha.
        uw_from_past_invest = self.unmanaged_wood_invest_before_year_start[
            'investment'] / uw_cost
        # unmanaged wood from actual invest
        uw_from_invest = self.unmanaged_wood_investment['investment'] / uw_cost
        # concat all unmanaged wood form invest
        uw_added = pd.concat([uw_from_past_invest, uw_from_invest])
        # remove value that exceed year_end
        for i in range(0, construction_delay):
            uw_added = np.delete(uw_added, len(uw_added) - 1)

        # Surface part
        self.unmanaged_wood_df['delta_surface'] = uw_added
        self.unmanaged_wood_df['cumulative_surface'] = np.cumsum(
            uw_added) + self.unmanaged_wood_initial_surface

        # Biomass production part
        ##### precise what is 3.6 ################
        self.unmanaged_wood_df['delta_biomass_production'] = self.unmanaged_wood_df['delta_surface'] * density_per_ha * mean_density * 3.6 / \
            years_between_harvest / (1 - recycle_part)
        self.unmanaged_wood_df['biomass_production'] = np.cumsum(
            self.unmanaged_wood_df['delta_biomass_production']) + self.unmanaged_wood_inital_prod
        self.unmanaged_wood_df['residues_production'] = self.unmanaged_wood_df['biomass_production'] * \
            residue_density_percentage * (1 - residue_percentage_for_energy)

        # CO2 part
        self.managed_wood_df['delta_CO2_emitted'] = - \
            self.managed_wood_df['delta_surface'] * self.CO2_per_ha
        self.managed_wood_df['CO2_emitted'] = - \
            self.managed_wood_df['cumulative_surface'] * self.CO2_per_ha

    def compute_reforestation_deforestation(self):
        """
        compute land use and due to reforestation et deforestation activities
        CO2 is not computed here because surface limit need to be taken into account before.
        """
        # forest surface is in Gha, deforestation_surface is in Mha,
        # deforested_surface is in Gha
        self.forest_surface_df['delta_deforestation_surface'] = - \
            self.deforestation_surface['deforested_surface'].values / 1000

        # forested surface
        # invest in G$, coest_per_ha in $/ha --> Gha
        self.forest_surface_df['delta_reforestation_surface'] = self.forest_investment['forest_investment'].values / self.cost_per_ha

        self.forest_surface_df['deforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_deforestation_surface'])
        self.forest_surface_df['reforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_reforestation_surface'])

    def sumup_global_surface_data(self):
        """
        managed wood and unmanaged wood impact forest_surface_df
        """
        self.forest_surface_df['delta_global_forest_surface'] = self.forest_surface_df['delta_reforestation_surface'] + self.forest_surface_df['delta_deforestation_surface'] +\
            self.unmanaged_wood_df['delta_surface'] + \
            self.managed_wood_df['delta_surface']
        self.forest_surface_df['global_forest_surface'] = self.forest_surface_df['reforestation_surface'] + self.forest_surface_df['deforestation_surface'] + \
            self.unmanaged_wood_df['cumulative_surface'] + \
            self.managed_wood_df['cumulative_surface']

    def check_deforestation_limit(self):
        """
        take into acount deforestation limit.
        If limit is not crossed, nothing happen
        If limit is crossed, deforestation_surface is limited and delta_deforestation is set to 0.
        """

        # check limit of deforestation
        for element in range(0, len(self.years)):
            if self.forest_surface_df.loc[element, 'global_forest_surface'] < -self.limit_deforestation_surface / 1000:
                self.forest_surface_df.loc[element,
                                           'delta_global_forest_surface'] = 0
                self.forest_surface_df.loc[element, 'delta_deforestation_surface'] = - \
                    self.forest_surface_df.loc[element,
                                               'delta_global_forest_surface']
                self.forest_surface_df.loc[element,
                                           'global_forest_surface'] = -self.limit_deforestation_surface / 1000
                self.forest_surface_df.loc[element,
                                           'deforestation_surface'] = -self.forest_surface_df.loc[element, 'reforestation_surface'] - self.managed_wood_df.loc[element, 'cumulative_surface'] - self.unmanaged_wood_df.loc[element, 'cumulative_surface'] - self.limit_deforestation_surface / 1000

    def compute_global_CO2_production(self):
        """
        compute the global CO2 production in Gt
        """
        # in Gt of CO2
        self.CO2_emitted_df['delta_CO2_emitted'] = -self.forest_surface_df['delta_global_forest_surface'] * \
            self.CO2_per_ha / 1000
        self.CO2_emitted_df['delta_CO2_deforestation'] = -self.forest_surface_df['delta_deforestation_surface'] * \
            self.CO2_per_ha / 1000
        self.CO2_emitted_df['delta_CO2_reforestation'] = -self.forest_surface_df['delta_reforestation_surface'] * \
            self.CO2_per_ha / 1000

        self.CO2_emitted_df['CO2_deforestation'] = -self.forest_surface_df['deforestation_surface'] * \
            self.CO2_per_ha / 1000
        self.CO2_emitted_df['CO2_reforestation'] = -self.forest_surface_df['reforestation_surface'] * \
            self.CO2_per_ha / 1000
        # global sum up
        self.CO2_emitted_df['global_CO2_emitted'] = -self.forest_surface_df['deforestation_surface'] * \
            self.CO2_per_ha / 1000 + self.initial_emissions
        self.CO2_emitted_df['global_CO2_captured'] = -(self.forest_surface_df['reforestation_surface'] +
                                                       self.unmanaged_wood_df['cumulative_surface'] +
                                                       self.managed_wood_df['cumulative_surface']) * self.CO2_per_ha / 1000
        self.CO2_emitted_df['global_CO2_emission_balance'] = self.CO2_emitted_df['global_CO2_emitted'] + \
            self.CO2_emitted_df['global_CO2_captured']

    # Gradients
    def d_deforestation_surface_d_deforestation_surface(self, ):
        """
        Compute gradient of deforestation surface by deforestation_surface (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_deforestation_surface_d_forests = np.identity(number_of_values)
        for i in range(0, number_of_values):
            if self.forest_surface_df.loc[i, 'forest_surface_evol_cumulative'] != -self.limit_deforestation_surface / 1000:
                d_deforestation_surface_d_forests[i][i] = - 1 / 1000
            else:
                d_deforestation_surface_d_forests[i][i] = 0

        return d_deforestation_surface_d_forests

    def d_forestation_surface_d_invest(self, ):
        """
        Compute gradient of deforestation surface by deforestation_surface (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_forestation_surface_d_invest = np.identity(number_of_values)
        for i in range(0, number_of_values):
            if self.forest_surface_df.loc[i, 'forest_surface_evol_cumulative'] != -self.limit_deforestation_surface / 1000:
                d_forestation_surface_d_invest[i][i] = 1 / self.cost_per_ha
            else:
                d_forestation_surface_d_invest[i][i] = 0

        return d_forestation_surface_d_invest

    def d_cum(self, derivative):
        """
        compute the gradient of a cumulative derivative
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_cum = np.identity(number_of_values)
        for i in range(0, number_of_values):
            d_cum[i] = derivative[i]
            if derivative[i][i] != 0:
                if i > 0:
                    d_cum[i] += d_cum[i - 1]
        return d_cum

    def d_CO2_emitted(self, d_deforestation_surface):
        """
        Compute gradient of non_captured_CO2 by deforestation surface
        :param: d_deforestation_surface, derivative of deforestation surface
        """

        d_CO2_emitted = - d_deforestation_surface * self.CO2_per_ha / 1000

        return d_CO2_emitted
