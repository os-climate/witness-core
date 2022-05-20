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
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.core.stream_type.carbon_models.carbon_dioxyde import CO2


class Forest():
    """
    Forest model class 
    basic for now, to evolve 

    """
    YEAR_START = 'year_start'
    YEAR_END = 'year_end'
    TIME_STEP = 'time_step'
    LIMIT_DEFORESTATION_SURFACE = 'limit_deforestation_surface'
    CO2_PER_HA = 'CO2_per_ha'
    INITIAL_CO2_EMISSIONS = 'initial_emissions'
    REFORESTATION_INVESTMENT = 'forest_investment'
    REFORESTATION_COST_PER_HA = 'reforestation_cost_per_ha'
    WOOD_TECHNO_DICT = 'wood_techno_dict'
    MW_INITIAL_PROD = 'managed_wood_initial_prod'
    MW_INITIAL_SURFACE = 'managed_wood_initial_surface'
    MW_INVEST_BEFORE_YEAR_START = 'managed_wood_invest_before_year_start'
    MW_INVESTMENT = 'managed_wood_investment'
    DEFORESTATION_INVESTMENT = 'deforestation_investment'
    DEFORESTATION_COST_PER_HA = 'deforestation_cost_per_ha'

    TRANSPORT_COST = 'transport_cost'
    MARGIN = 'margin'
    UNMANAGED_FOREST = 'initial_unmanaged_forest_surface'
    PROTECTED_FOREST = 'protected_forest_surface'

    FOREST_SURFACE_DF = 'forest_surface_df'
    FOREST_DETAIL_SURFACE_DF = 'forest_surface_detail_df'
    CO2_EMITTED_FOREST_DF = 'CO2_land_emissions'
    CO2_EMITTED_DETAIL_DF = 'CO2_emissions_detail_df'
    MW_DF = 'managed_wood_df'
    #UW_DF = 'unmanaged_wood_df'
    BIOMASS_DRY_DETAIL_DF = 'biomass_dry_detail_df'
    BIOMASS_DRY_DF = 'biomass_dry_df'

    def __init__(self, param):
        """
        Constructor
        """
        self.param = param
        self.product_energy_unit = 'TWh'
        self.mass_unit = 'Mt'
        self.set_data()
        self.create_dataframe()
        self.counter = 0

    def set_data(self):
        """
        """
        self.year_start = self.param[self.YEAR_START]
        self.year_end = self.param[self.YEAR_END]
        self.time_step = self.param[self.TIME_STEP]
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]
        #self.deforestation_surface = self.param[self.DEFORESTATION_SURFACE]
        self.CO2_per_ha = self.param[self.CO2_PER_HA]
        # initial CO2 emissions
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        # forest data
        self.forest_investment = self.param[self.REFORESTATION_INVESTMENT]
        self.cost_per_ha = self.param[self.REFORESTATION_COST_PER_HA]
        self.deforest_invest = self.param[self.DEFORESTATION_INVESTMENT]
        self.deforest_cost_per_ha = self.param[self.DEFORESTATION_COST_PER_HA]
        self.techno_wood_info = self.param[self.WOOD_TECHNO_DICT]
        self.managed_wood_inital_prod = self.param[self.MW_INITIAL_PROD]
        self.managed_wood_initial_surface = self.param[self.MW_INITIAL_SURFACE]
        self.managed_wood_invest_before_year_start = self.param[
            self.MW_INVEST_BEFORE_YEAR_START]
        self.managed_wood_investment = self.param[self.MW_INVESTMENT]
        self.transport = self.param[self.TRANSPORT_COST]
        self.margin = self.param[self.MARGIN]
        self.initial_unmanaged_forest_surface = self.param[self.UNMANAGED_FOREST]
        self.protected_forest_surface = self.param[self.PROTECTED_FOREST]
        self.scaling_factor_techno_consumption = self.param['scaling_factor_techno_consumption']
        self.scaling_factor_techno_production = self.param['scaling_factor_techno_production']

    def create_dataframe(self):
        """
        Create the dataframe and fill it with values at year_start
        """
        years = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
        self.years = years
        self.forest_surface_df = pd.DataFrame({'years': self.years})
        self.CO2_emitted_df = pd.DataFrame({'years': self.years})
        self.managed_wood_df = pd.DataFrame({'years': self.years})
        #self.unmanaged_wood_df = pd.DataFrame({'years': self.years})
        self.biomass_dry_df = pd.DataFrame({'years': self.years})
        self.ratio = pd.DataFrame({'years': self.years})

        # output dataframes:
        self.techno_production = pd.DataFrame({'years': self.years})
        self.techno_prices = pd.DataFrame({'years': self.years})
        self.techno_consumption = pd.DataFrame({'years': self.years})
        self.techno_consumption_woratio = pd.DataFrame({'years': self.years})
        self.land_use_required = pd.DataFrame({'years': self.years})
        self.CO2_emissions = pd.DataFrame({'years': self.years})
        self.lost_capital = pd.DataFrame({'years': self.years})
        self.techno_capital = pd.DataFrame({'years': self.years})

    def compute(self, in_dict):
        """
        Computation methods
        """
        self.biomass_dry_calorific_value = BiomassDry.data_energy_dict[
            'calorific_value']  # kwh/kg
        self.biomass_dry_high_calorific_value = BiomassDry.data_energy_dict[
            'high_calorific_value']  # kwh/kg
        self.year_start = in_dict[self.YEAR_START]
        self.year_end = in_dict[self.YEAR_END]
        self.time_step = in_dict[self.TIME_STEP]
        self.forest_investment = in_dict[self.REFORESTATION_INVESTMENT]
        self.deforest_invest = in_dict[self.DEFORESTATION_INVESTMENT]
        self.cost_per_ha = in_dict[self.REFORESTATION_COST_PER_HA]
        self.initial_emissions = self.param[self.INITIAL_CO2_EMISSIONS]
        self.limit_deforestation_surface = self.param[self.LIMIT_DEFORESTATION_SURFACE]
        self.years = np.arange(
            self.year_start, self.year_end + 1, self.time_step)
        self.managed_wood_investment = in_dict[self.MW_INVESTMENT]

        self.forest_surface_df['unmanaged_forest'] = self.initial_unmanaged_forest_surface
        # compute data of each contribution
        self.compute_managed_wood_surface()
        self.compute_reforestation_deforestation()
        self.compute_managed_wood_production()
        # sum up global surface data
        self.sumup_global_surface_data()
        # check deforestation limit
        self.check_deforestation_limit()
        # compute capital and lost capital
        self.compute_lost_capital()
        # sum up global CO2 data
        self.compute_global_CO2_production()

        # compute biomass dry production
        self.compute_biomass_dry_production()

        # compute outputs:

        # compute land_use for energy
        self.land_use_required['Forest (Gha)'] = self.managed_wood_df['cumulative_surface']

        # compute forest constrain evolution: reforestation + deforestation
        self.forest_surface_df['forest_constraint_evolution'] = self.forest_surface_df['reforestation_surface'] + \
            self.forest_surface_df['deforestation_surface']

        # techno production in TWh
        self.techno_production[f'{BiomassDry.name} ({BiomassDry.unit})'] = self.biomass_dry_df[
            'biomass_dry_for_energy (Mt)'] * self.biomass_dry_calorific_value
        # price in $/MWh
        self.techno_prices['Forest'] = self.biomass_dry_df['price_per_MWh']

        if 'CO2_taxes_factory' in self.biomass_dry_df:
            self.techno_prices['Forest_wotaxes'] = self.biomass_dry_df['price_per_MWh'] - \
                self.biomass_dry_df['CO2_taxes_factory']
        else:
            self.techno_prices['Forest_wotaxes'] = self.biomass_dry_df['price_per_MWh']

        # CO2 emissions
        self.compute_carbon_emissions()

        # CO2 consumed
        self.techno_consumption[f'{CO2.name} ({self.mass_unit})'] = -self.techno_wood_info['CO2_from_production'] / \
            self.biomass_dry_high_calorific_value * \
            self.techno_production[f'{BiomassDry.name} ({BiomassDry.unit})']

        self.techno_consumption_woratio[f'{CO2.name} ({self.mass_unit})'] = -self.techno_wood_info['CO2_from_production'] / \
            self.biomass_dry_high_calorific_value * \
            self.techno_production[f'{BiomassDry.name} ({BiomassDry.unit})']

    def compute_managed_wood_surface(self):
        """
        """
        construction_delay = self.techno_wood_info['construction_delay']
        mw_cost = self.techno_wood_info['managed_wood_price_per_ha']
        # managed wood from past invest. invest in G$ - surface in Gha.
        mw_from_past_invest = self.managed_wood_invest_before_year_start[
            'investment'] / mw_cost
        # managed wood from actual invest
        mw_from_invest = self.managed_wood_investment['investment'] / mw_cost
        # concat all managed wood form invest
        mw_added = pd.concat([mw_from_past_invest, mw_from_invest]).values

        # remove value that exceed year_end
        for i in range(0, construction_delay):
            mw_added = np.delete(mw_added, len(mw_added) - 1)

        self.managed_wood_df['delta_surface'] = mw_added
        cumulative_mw = np.cumsum(mw_added)
        self.managed_wood_df['cumulative_surface'] = cumulative_mw + \
            self.managed_wood_initial_surface
        self.forest_surface_df['unmanaged_forest'] = self.forest_surface_df['unmanaged_forest'] - cumulative_mw

    def compute_managed_wood_production(self):
        """
        compute data concerning managed wood : surface taken, production, CO2 absorbed, as delta and cumulative
        """
        density_per_ha = self.techno_wood_info['density_per_ha']
        mean_density = self.techno_wood_info['density']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        residue_density_percentage = self.techno_wood_info['residue_density_percentage']
        residue_percentage_for_energy = self.techno_wood_info['residue_percentage_for_energy']
        wood_percentage_for_energy = self.techno_wood_info['wood_percentage_for_energy']

        # Biomass production part
        # Gha * m3/ha * kg/m3 => Mt
        # recycle part is from the 2nd hand wood that will be recycled from the
        # first investment
        self.managed_wood_df['delta_biomass_production (Mt)'] = self.managed_wood_df['delta_surface'] * density_per_ha * mean_density / \
            years_between_harvest / (1 - recycle_part)
        self.managed_wood_df['biomass_production (Mt)'] = np.cumsum(
            self.managed_wood_df['delta_biomass_production (Mt)']) + self.managed_wood_inital_prod / self.biomass_dry_calorific_value
        self.managed_wood_df['residues_production (Mt)'] = self.managed_wood_df['biomass_production (Mt)'] * \
            residue_density_percentage
        self.managed_wood_df['residues_production_for_energy (Mt)'] = self.managed_wood_df['residues_production (Mt)'] * \
            residue_percentage_for_energy
        self.managed_wood_df['residues_production_for_industry (Mt)'] = self.managed_wood_df['residues_production (Mt)'] * \
            (1 - residue_percentage_for_energy)

        self.managed_wood_df['wood_production (Mt)'] = self.managed_wood_df['biomass_production (Mt)'] * \
            (1 - residue_density_percentage)
        self.managed_wood_df['wood_production_for_energy (Mt)'] = self.managed_wood_df['wood_production (Mt)'] * \
            wood_percentage_for_energy
        self.managed_wood_df['wood_production_for_industry (Mt)'] = self.managed_wood_df['wood_production (Mt)'] * \
            (1 - wood_percentage_for_energy)

        # CO2 part
        self.managed_wood_df['delta_CO2_emitted'] = - \
            self.managed_wood_df['delta_surface'] * self.CO2_per_ha / 1000
        # CO2 emitted is delta cumulate
        self.managed_wood_df['CO2_emitted'] = - \
            (self.managed_wood_df['cumulative_surface'] -
             self.managed_wood_initial_surface) * self.CO2_per_ha / 1000

    def compute_reforestation_deforestation(self):
        """
        compute land use and due to reforestation et deforestation activities
        CO2 is not computed here because surface limit need to be taken into account before.
        """
        density_per_ha = self.techno_wood_info['density_per_ha']
        mean_density = self.techno_wood_info['density']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        wood_percentage_for_energy = self.techno_wood_info['wood_percentage_for_energy']
        # forest surface is in Gha, deforestation_surface is in Mha,
        # deforested_surface is in Gha
        self.forest_surface_df['delta_deforestation_surface'] = - \
            self.deforest_invest['investment'].values / \
            self.deforest_cost_per_ha

        # forested surface
        # invest in G$, coest_per_ha in $/ha --> Gha
        self.forest_surface_df['delta_reforestation_surface'] = self.forest_investment['forest_investment'].values / self.cost_per_ha

        self.forest_surface_df['deforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_deforestation_surface'])
        self.forest_surface_df['reforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_reforestation_surface'])
        self.forest_surface_df['unmanaged_forest'] += self.forest_surface_df['reforestation_surface'] + \
            self.forest_surface_df['deforestation_surface']

        for i in range(0, len(self.years)):
            # if unmanaged forest are empty, managed forest are removed
            if self.forest_surface_df.loc[i, 'unmanaged_forest'] <= 0:
                self.managed_wood_df.loc[i,
                                         'delta_surface'] += self.forest_surface_df.loc[i, 'unmanaged_forest']
                self.managed_wood_df.loc[i,
                                         'cumulative_surface'] += self.forest_surface_df.loc[i, 'unmanaged_forest']

                self.managed_wood_df.loc[i, 'delta_surface'] = max(
                    self.managed_wood_df.loc[i, 'delta_surface'], -self.managed_wood_df.loc[i, 'cumulative_surface'])
                self.forest_surface_df.loc[i, 'unmanaged_forest'] = 0
                # if managed forest are empty, nothing is removed
                if self.managed_wood_df.loc[i, 'cumulative_surface'] <= 0:

                    sum = np.cumsum(self.managed_wood_df['delta_surface'])
                    # delta is all the managed wood available
                    self.managed_wood_df.loc[i,
                                             'delta_surface'] = -(sum[i - 1] + self.managed_wood_initial_surface)
                    self.managed_wood_df.loc[i, 'cumulative_surface'] = 0
                    self.forest_surface_df.loc[i, 'delta_deforestation_surface'] = - \
                        self.forest_surface_df.loc[i,
                                                   'delta_reforestation_surface']

        self.forest_surface_df['deforestation_surface'] = np.cumsum(
            self.forest_surface_df['delta_deforestation_surface'])

        self.biomass_dry_df['deforestation (Mt)'] = -self.forest_surface_df['delta_deforestation_surface'] * \
            density_per_ha * mean_density / (1 - recycle_part)
        self.biomass_dry_df['deforestation_for_energy'] = self.biomass_dry_df['deforestation (Mt)'] * \
            wood_percentage_for_energy
        self.biomass_dry_df['deforestation_for_industry'] = self.biomass_dry_df['deforestation (Mt)'] - \
            self.biomass_dry_df['deforestation_for_energy']
        self.biomass_dry_df['deforestation_price_per_ton'] = density_per_ha * mean_density / \
            (1 - recycle_part) / self.deforest_cost_per_ha
        self.biomass_dry_df['deforestation_price_per_MWh'] = self.biomass_dry_df['deforestation_price_per_ton'] / \
            self.biomass_dry_calorific_value

    def sumup_global_surface_data(self):
        """
        managed wood and unmanaged wood impact forest_surface_df
        """
        self.forest_surface_df['delta_global_forest_surface'] = self.forest_surface_df['delta_reforestation_surface'] + \
            self.forest_surface_df['delta_deforestation_surface']
        self.forest_surface_df['global_forest_surface'] = self.managed_wood_df['cumulative_surface'] + \
            self.forest_surface_df['unmanaged_forest'] + \
            self.protected_forest_surface
        self.forest_surface_df['protected_forest_surface'] = self.protected_forest_surface

    def check_deforestation_limit(self):
        """
        take into acount deforestation limit.
        If limit is not crossed, nothing happen
        If limit is crossed, deforestation_surface is limited and delta_deforestation is set to 0.
        """

        # check limit of deforestation
        for i in range(0, len(self.years)):
            if self.forest_surface_df.loc[i, 'global_forest_surface'] < -self.limit_deforestation_surface / 1000:
                self.forest_surface_df.loc[i,
                                           'delta_global_forest_surface'] = 0
                self.forest_surface_df.loc[i, 'delta_deforestation_surface'] = - \
                    self.forest_surface_df.loc[i,
                                               'delta_global_forest_surface']
                self.forest_surface_df.loc[i,
                                           'global_forest_surface'] = -self.limit_deforestation_surface / 1000
                self.forest_surface_df.loc[i, 'deforestation_surface'] = - self.forest_surface_df.loc[i, 'reforestation_surface'] - \
                    self.managed_wood_df.loc[i, 'cumulative_surface'] - \
                    self.limit_deforestation_surface / 1000

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
            self.CO2_per_ha / 1000 + self.initial_emissions
        self.CO2_emitted_df['CO2_reforestation'] = -self.forest_surface_df['reforestation_surface'] * \
            self.CO2_per_ha / 1000
        # global sum up
        self.CO2_emitted_df['global_CO2_emitted'] = -self.forest_surface_df['deforestation_surface'] * \
            self.CO2_per_ha / 1000 + self.initial_emissions
        self.CO2_emitted_df['global_CO2_captured'] = -self.forest_surface_df['reforestation_surface'] * \
            self.CO2_per_ha / 1000
        self.CO2_emitted_df['emitted_CO2_evol_cumulative'] = self.CO2_emitted_df['global_CO2_emitted'] + \
            self.CO2_emitted_df['global_CO2_captured']

    def compute_biomass_dry_production(self):
        """
        compute total biomass dry prod
        """

        self.biomass_dry_df['biomass_dry_for_energy (Mt)'] = self.managed_wood_df['wood_production_for_energy (Mt)'] + \
            self.managed_wood_df['residues_production_for_energy (Mt)'] + \
            self.biomass_dry_df['deforestation_for_energy']

        self.compute_price('managed_wood')

        self.managed_wood_part = self.managed_wood_df['biomass_production (Mt)'] / (
            self.managed_wood_df['biomass_production (Mt)'] + self.biomass_dry_df['deforestation (Mt)'])
        self.deforestation_part = self.biomass_dry_df['deforestation (Mt)'] / (
            self.managed_wood_df['biomass_production (Mt)'] + self.biomass_dry_df['deforestation (Mt)'])

        self.biomass_dry_df['price_per_ton'] = self.biomass_dry_df['managed_wood_price_per_ton'] * self.managed_wood_part + \
            self.biomass_dry_df['deforestation_price_per_ton'] * \
            self.deforestation_part

        self.biomass_dry_df['managed_wood_price_per_MWh'] = self.biomass_dry_df['managed_wood_price_per_ton'] / \
            self.biomass_dry_calorific_value
        self.biomass_dry_df['price_per_MWh'] = self.biomass_dry_df['price_per_ton'] / \
            self.biomass_dry_calorific_value

    def compute_price(self, techno_name):
        """
        compute price as in techno_type
        """

        # Maximize with smooth exponential
#         price_df['invest'] = compute_func_with_exp_min(
#             investment, self.min_value_invest)
        density_per_ha = self.techno_wood_info['density_per_ha']  # m3/ha
        mean_density = self.techno_wood_info['density']  # kg/m3

        self.crf = self.compute_crf()

        self.biomass_dry_df[f'{techno_name}_transport ($/t)'] = self.transport['transport']

        # Factory cost including CAPEX OPEX
        # $/ha * ha/m3 * m3/kg * 1000 = $/t
        self.biomass_dry_df[f'{techno_name}_capex ($/t)'] = self.techno_wood_info[f'{techno_name}_price_per_ha'] * \
            (self.crf + 0.045) / density_per_ha / mean_density * 1000

        self.biomass_dry_df[f'{techno_name}_price_per_ton'] = (
            self.biomass_dry_df[f'{techno_name}_capex ($/t)'] +
            self.biomass_dry_df[f'{techno_name}_transport ($/t)']) * self.margin['margin'] / 100.0

    def compute_crf(self):
        """
        Compute annuity factor with the Weighted averaged cost of capital
        and the lifetime of the selected solution
        """
        wacc = self.techno_wood_info['WACC']
        crf = (wacc * (1.0 + wacc) ** 100) / \
              ((1.0 + wacc) ** 100 - 1.0)

        return crf

    def compute_lost_capital(self):
        """
        Compute the loss of capital due to reforestation and deforestation activities that have opposite effect but cost money.
        To deforest and to reforest only for surface expanse result as a lost of capital.

        lost_capital = min(reforest_surface, deforest_surface) * cost_per_ha
        cost_per_ha is in $/ha
        reforest_surface and deforest_surface are in Gha
        lost_capital is in G$ 
        """
        self.lost_capital['Forest'] = 0
        self.techno_capital['Forest'] = 0
#         self.lost_capital['Deforestation'] = 0
#         self.techno_capital['Deforestation'] = 0
        # abs() needed because deforestation surface is negative

        for element in range(0, len(self.years)):
            if abs(self.forest_surface_df.at[element, 'delta_deforestation_surface']) < self.forest_surface_df.at[element, 'delta_reforestation_surface']:
                self.lost_capital.loc[element, 'Forest'] = abs(self.forest_surface_df.loc[element,
                                                                                          'delta_deforestation_surface']) * self.cost_per_ha
            else:
                self.lost_capital.loc[element, 'Forest'] = self.forest_surface_df.loc[element,
                                                                                      'delta_reforestation_surface'] * self.cost_per_ha

        self.techno_capital['Forest'] = self.forest_surface_df['delta_reforestation_surface'] * self.cost_per_ha
#         self.techno_capital['Managed_wood'] = self.managed_wood_investment['investment']
#         self.lost_capital['Managed_wood'] = self.managed_wood_investment['investment'] * \
#             (1 - np.array(self.ratio))

    def compute_carbon_emissions(self):
        '''
        Compute the carbon emissions from the technology taking into account 
        CO2 from production + CO2 from primary resources 
        '''
        if 'CO2_from_production' not in self.techno_wood_info:
            self.CO2_emissions['production'] = self.get_theoretical_co2_prod(
                unit='kg/kWh')
        elif self.techno_wood_info['CO2_from_production'] == 0.0:
            self.CO2_emissions['production'] = 0.0
        else:
            if self.techno_wood_info['CO2_from_production_unit'] == 'kg/kg':
                self.CO2_emissions['production'] = self.techno_wood_info['CO2_from_production'] / \
                    self.biomass_dry_high_calorific_value
            elif self.techno_wood_info['CO2_from_production_unit'] == 'kg/kWh':
                self.CO2_emissions['production'] = self.techno_wood_info['CO2_from_production']

        # Add carbon emission from input energies (resources or other
        # energies)

        co2_emissions_frominput_energies = self.compute_CO2_emissions_from_input_resources(
        )

        # Add CO2 from production + C02 from input energies
        self.CO2_emissions['Forest'] = self.CO2_emissions['production'] + \
            co2_emissions_frominput_energies

    def get_theoretical_co2_prod(self, unit='kg/kWh'):
        ''' 
        Get the theoretical CO2 production for a given technology,
        '''
        return 0.0

    def compute_CO2_emissions_from_input_resources(self):
        '''
        Need to take into account  CO2 from electricity/fuel production
        '''
        return 0.0

    # Gradients
    def d_deforestation_surface_d_deforestation_invest(self, ):
        """
        Compute gradient of deforestation surface by deforestation_invest (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_deforestation_surface_d_forests = np.identity(number_of_values)
        for i in range(0, number_of_values):
            # derivate = -1/1000 for unit conversion if limit is not broken
            if self.forest_surface_df['global_forest_surface'].values[i] != -self.limit_deforestation_surface / 1000:
                d_deforestation_surface_d_forests[i][i] = - \
                    1 / self.deforest_cost_per_ha
            # if limit is broken, grad is null
            else:
                d_deforestation_surface_d_forests[i][i] = 0

        return d_deforestation_surface_d_forests
    # alternative

    def d_forestation_surface_d_invest(self, ):
        """
        Compute gradient of reforestation surface by invest (design variable)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_forestation_surface_d_invest = np.identity(number_of_values)
        for i in range(0, number_of_values):
            # surface = invest / cost_per_ha if limit is not borken
            d_forestation_surface_d_invest[i][i] = 1 / self.cost_per_ha
#             if self.forest_surface_df['global_forest_surface'].values[i] != -self.limit_deforestation_surface / 1000:
#                 d_forestation_surface_d_invest[i][i] = 1 / self.cost_per_ha
#             #surface = constant is limit is broken
#             else:
#                 d_forestation_surface_d_invest[i][i] = 0

#     def d_forestation_surface_d_invest(self, ):
#         """
#         Compute gradient of reforestation surface by invest (design variable)
#         """
#         number_of_values = (self.year_end - self.year_start + 1)
#         d_forestation_surface_d_invest = np.identity(number_of_values)
#         for i in range(0, number_of_values):
#             # surface = invest / cost_per_ha if limit is not borken
#             d_forestation_surface_d_invest[i][i] = 1 / self.cost_per_ha
# #             if self.forest_surface_df['global_forest_surface'].values[i] != -self.limit_deforestation_surface / 1000:
# #                 d_forestation_surface_d_invest[i][i] = 1 / self.cost_per_ha
# #             #surface = constant is limit is broken
# #             else:
# #                 d_forestation_surface_d_invest[i][i] = 0

        return d_forestation_surface_d_invest

    def d_wood_techno_surface_d_invest(self, price_per_ha):
        """
        Compute gradient of managed wood surface by invest
        Same function for managed wood and unmanaged wood. Only the price_per_ha change.
        construction delay impact becasue there is a shift of investment impact of construction_delay year.
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_wood_surface_d_invest = np.identity(number_of_values) * 0
        construction_delay = self.techno_wood_info['construction_delay']
        for i in range(construction_delay, number_of_values):
            if self.forest_surface_df.loc[i, 'unmanaged_forest'] <= 0:
                d_wood_surface_d_invest[i][i -
                                           construction_delay] = 1 / price_per_ha
            else:
                d_wood_surface_d_invest[i][i -
                                           construction_delay] = 1 / price_per_ha

        return d_wood_surface_d_invest

    def d_managed_wood_surf_d_invest_reforestation(self, d_forestation_surface_d_invest):
        """
        in the case unmanaged_forest are null, the evolution of managed wood is limited by the evolution of unmanaged forest
        as a result, managed wood evolve as reforestation and deforestation.
        """

        number_of_values = (self.year_end - self.year_start + 1)
        d_wood_surface_d_invest = np.identity(number_of_values) * 0
        for i in range(0, number_of_values):
            if self.forest_surface_df.loc[i, 'unmanaged_forest'] <= 0:
                self.counter += 1
                for j in range(0, i + 1):
                    d_wood_surface_d_invest[i][j] = d_forestation_surface_d_invest[i][i]

        return d_wood_surface_d_invest

    def d_cum(self, derivative):
        """
        compute the gradient of a cumulative derivative
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_cum = np.identity(number_of_values)
        for i in range(0, number_of_values):
            d_cum[i] = derivative[i]
            if i > 0:
                d_cum[i] += d_cum[i - 1]
        return d_cum

    def d_cum_managed_forest(self, derivative):
        """
        compute the gradient of a cumulative derivative
        a special function for managed forest is needed due to unmanaged_forest limitation
        """
        number_of_values = (self.year_end - self.year_start + 1)
        d_cum = np.identity(number_of_values) * 0
        for i in range(0, number_of_values):
            if self.ratio.loc[i, 'ratio'] < 1:
                pass
            else:
                d_cum[i] = derivative[i]
                if i > 0:
                    d_cum[i] += d_cum[i - 1]
        return d_cum

    def d_CO2_emitted(self, d_deforestation_surface):
        """
        Compute gradient of non_captured_CO2 by deforestation surface
        :param: d_deforestation_surface, derivative of deforestation surface
        CO2_emitted = surface * constant --> d_surface is reused.
        """

        d_CO2_emitted = - d_deforestation_surface * self.CO2_per_ha / 1000

        return d_CO2_emitted

    def d_biomass_prod_d_invest(self, d_surf_d_invest, wood_or_residues_percentage, percentage_for_energy):
        """
        Compute derivate of biomass production by investment. Biomass production is : mw_residu / un_residu / mw_wood / uw_wood
        prod = surface * density_per_ha * density * wood_or_residues_percentage * percentage_for_energy / years_between_harvest / (1 - recycle_part)
        --> only surface is dependant of invest, the other parameters does not depends of invest.
        d_surf_d_invest is alread computed and known.
        # recycle part is from the 2nd hand wood that will be recycled from the first investment
        """
        number_of_values = (self.year_end - self.year_start + 1)
        density = self.techno_wood_info['density']
        density_per_ha = self.techno_wood_info['density_per_ha']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']

        ddelta_prod_dinvest = d_surf_d_invest * density_per_ha * density * \
            wood_or_residues_percentage * percentage_for_energy / \
            years_between_harvest / \
            (1 - recycle_part)

        return ddelta_prod_dinvest

    def d_biomass_prod_d_invest_reforestation(self, d_surf_d_invest, wood_or_residues_percentage, percentage_for_energy):
        """
        Compute derivate of biomass production by investment. Biomass production is : mw_residu / un_residu / mw_wood / uw_wood
        prod = surface * density_per_ha * density * wood_or_residues_percentage * percentage_for_energy / years_between_harvest / (1 - recycle_part)
        --> only surface is dependant of invest, the other parameters does not depends of invest.
        d_surf_d_invest is alread computed and known.
        # recycle part is from the 2nd hand wood that will be recycled from the first investment
        """
        density = self.techno_wood_info['density']
        density_per_ha = self.techno_wood_info['density_per_ha']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        number_of_values = (self.year_end - self.year_start + 1)
        years_of_impact = number_of_values - self.counter

        ddelta_prod_dinvest = d_surf_d_invest * density_per_ha * density * \
            wood_or_residues_percentage * percentage_for_energy / \
            years_between_harvest / \
            (1 - recycle_part)
#         for i in range(years_of_impact, number_of_values):
#             for j in range(1, i + 1 - years_of_impact):
#                 ddelta_prod_dinvest[i][years_of_impact + j] = 0

        return ddelta_prod_dinvest

    def d_biomass_price_d_invest_mw(self, price_per_ha):
        """
        compute derivate of biomass price by invest in managed wood
        price = mw_price * mw_part + deforest_price * deforest_part
        mw_price and deforest_price are independant of invest
        mw_part = mw_prod / (mw_prod + deforest_prod) with mw_prod dependant of invest
        --> (u/v)' = (u'v - uv') / v^2
        and deforest_part = (1-mw_part)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        construction_delay = self.techno_wood_info['construction_delay']
        d_wood_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        res = np.zeros((number_of_values, number_of_values))
        for i in range(0, number_of_values):
            d_wood_surface_d_invest[i][i] = 1 / price_per_ha
        deriv_2 = self.d_cum(d_wood_surface_d_invest)
        d_surf_d_invest = deriv_2

        density = self.techno_wood_info['density']
        density_per_ha = self.techno_wood_info['density_per_ha']
        years_between_harvest = self.techno_wood_info['years_between_harvest']
        recycle_part = self.techno_wood_info['recycle_part']
        # compute d_prod_dinvest. total production is taken into account :
        # energy + non energy
        dprod_dinvest = d_surf_d_invest * density_per_ha * density / \
            years_between_harvest / \
            (1 - recycle_part)

        mw_prod = self.managed_wood_df['biomass_production (Mt)'].values
        biomass_prod = self.managed_wood_df['biomass_production (Mt)'].values + \
            self.biomass_dry_df['deforestation (Mt)'].values
        # (u/v)' = (u'v - uv') / v^2
        d_mwpart_d_mw_invest = (dprod_dinvest * biomass_prod - mw_prod *
                                dprod_dinvest) / biomass_prod**2 / self.biomass_dry_calorific_value

        derivate = self.biomass_dry_df['managed_wood_price_per_ton'].values * d_mwpart_d_mw_invest - \
            self.biomass_dry_df['deforestation_price_per_ton'].values * \
            d_mwpart_d_mw_invest
        # shift needed due to construction delay
        for i in range(construction_delay, number_of_values):
            for j in range(construction_delay, i + 1):
                res[i, j - construction_delay] = derivate[i, i]

        for i in range(0, number_of_values):
            if self.ratio.loc[i, 'ratio'] < 1:
                res[i] = 0

        return res

    def d_biomass_price_d_invest_reforestation(self, price_per_ha, dprod_dinvest):
        """
        compute derivate of biomass price by invest in managed wood
        price = mw_price * mw_part + deforest_price * deforest_part
        mw_price and deforest_price are independant of invest
        mw_part = mw_prod / (mw_prod + deforest_prod) with mw_prod dependant of invest
        --> (u/v)' = (u'v - uv') / v^2
        and deforest_part = (1-mw_part)
        """
        number_of_values = (self.year_end - self.year_start + 1)
        construction_delay = self.techno_wood_info['construction_delay']
        d_wood_surface_d_invest = np.zeros(
            (number_of_values, number_of_values))
        res = np.zeros((number_of_values, number_of_values))
        mw_prod = self.managed_wood_df['biomass_production (Mt)'].values
        biomass_prod = self.managed_wood_df['biomass_production (Mt)'].values + \
            self.biomass_dry_df['deforestation (Mt)'].values
        # (u/v)' = (u'v - uv') / v^2
        d_mwpart_d_ref_invest = (dprod_dinvest * biomass_prod - mw_prod *
                                 dprod_dinvest) / biomass_prod**2
        # *1e6 to go from MWh to TWh
        derivate = (self.biomass_dry_df['managed_wood_price_per_MWh'].values * d_mwpart_d_ref_invest -
                    self.biomass_dry_df['deforestation_price_per_MWh'].values *
                    d_mwpart_d_ref_invest)
        # shift needed due to construction delay
#         for i in range(construction_delay, number_of_values):
#             for j in range(construction_delay, i + 1):
#                 #                 if self.forest_surface_df.loc[i, 'unmanaged_forest'] <= 0:
#                 #                     pass
#                 #                 else:
#                 res[i, j - construction_delay] = derivate[i, i]
        return derivate

    def d_biomass_price_d_invest_deforest(self, dprod_dinvest_deforest):
        """
        compute derivate of biomass price by invest in deforestation
        price = mw_price * mw_part + deforest_price * deforest_part
        mw_price and deforest_price are independant of invest
        deforest_part = deforest_prod / (mw_prod + deforest_prod) with deforest_prod dependant of invest
        --> (u/v)' = (u'v - uv') / v^2
        and deforest_part = (1-mw_part)
        """
        wood_percentage_for_energy = self.techno_wood_info['wood_percentage_for_energy']

        dprod_dinvest = dprod_dinvest_deforest / wood_percentage_for_energy

        deforest_prod = self.biomass_dry_df['deforestation (Mt)'].values
        biomass_prod = self.managed_wood_df['biomass_production (Mt)'] .values + \
            self.biomass_dry_df['deforestation (Mt)'].values
        # (u/v)' = (u'v - uv') / v^2
        d_deforestpart_d_deforest_invest = (
            dprod_dinvest * biomass_prod - deforest_prod * dprod_dinvest) / biomass_prod**2 / self.biomass_dry_calorific_value

        # final result : price = price_def * def_part + mw_price * (1-def_part)
        derivate = self.biomass_dry_df['deforestation_price_per_ton'].values * d_deforestpart_d_deforest_invest - \
            self.biomass_dry_df['managed_wood_price_per_ton'].values * \
            d_deforestpart_d_deforest_invest

        return derivate

    def d_biomass_price_d_invest_deforest_limit(self, dprod_dinvest_deforest, dmwprod_dinvest_deforest):
        """
        compute derivate of biomass price by invest in deforestation
        price = mw_price * mw_part + deforest_price * deforest_part
        mw_price and deforest_price are independant of invest
        deforest_part = deforest_prod / (mw_prod + deforest_prod) with deforest_prod dependant of invest
        --> (u/v)' = (u'v - uv') / v^2
        and deforest_part = (1-mw_part)
        """
        wood_percentage_for_energy = self.techno_wood_info['wood_percentage_for_energy']

        dprod_dinvest = dprod_dinvest_deforest / wood_percentage_for_energy

        deforest_prod = self.biomass_dry_df['deforestation (Mt)'].values
        biomass_prod = self.managed_wood_df['biomass_production (Mt)'].values + \
            self.biomass_dry_df['deforestation (Mt)'].values
        # (u/v)' = (u'v - uv') / v^2
        d_deforestpart_d_deforest_invest = (
            (dprod_dinvest) * biomass_prod - deforest_prod * (dprod_dinvest + dmwprod_dinvest_deforest)) / biomass_prod**2 / self.biomass_dry_calorific_value

        d_mwpart_d_deforest_invest = (
            dprod_dinvest * biomass_prod - self.managed_wood_df['biomass_production (Mt)'].values * (dprod_dinvest + dmwprod_dinvest_deforest)) / biomass_prod**2 / self.biomass_dry_calorific_value

        # final result : price = price_def * def_part + mw_price * (1-def_part)
        derivate = (self.biomass_dry_df['deforestation_price_per_ton'].values * d_deforestpart_d_deforest_invest -
                    self.biomass_dry_df['managed_wood_price_per_ton'].values *
                    d_deforestpart_d_deforest_invest) / self.ratio['ratio'].values
#         derivate = (self.biomass_dry_df['deforestation_price_per_ton'].values * d_deforestpart_d_deforest_invest +
#                     self.biomass_dry_df['managed_wood_price_per_ton'].values *
#                     d_mwpart_d_deforest_invest)

        return derivate

    def d_biomass_prod_d_deforestation_invest(self, d_deforest_surf_d_deforest_invest):
        """
        Compute derivate of biomass prod by deforestation surface
        -self.forest_surface_df['delta_deforestation_surface'] * density_per_ha * mean_density / \
            years_between_harvest / (1 - recycle_part)
        """
        density_per_ha = self.techno_wood_info['density_per_ha']
        recycle_part = self.techno_wood_info['recycle_part']
        mean_density = self.techno_wood_info['density']
        wood_percentage_for_energy = self.techno_wood_info['wood_percentage_for_energy']
        result = -d_deforest_surf_d_deforest_invest * (density_per_ha) * mean_density / \
            (1 - recycle_part) * wood_percentage_for_energy
        return result

    def d_mw_surf_d_deforest_invest(self,):
        """
        """
        number_of_values = (self.year_end - self.year_start + 1)
        result = np.identity(number_of_values) * 0
        for i in range(0, number_of_values):
            if self.ratio.loc[i, 'ratio'] < 1:
                for j in range(0, i + 1):
                    result[i][j] = -1 / self.deforest_cost_per_ha

        return result

    def d_capital_total_d_invest(self,):
        """
        Compute derivate of capital total of reforestation regarding reforestation_investment
        """
        dcapital_d_invest = np.identity(len(self.years))

        return dcapital_d_invest

    def d_lostcapitald_invest(self, d_delta_reforestation_dinvest):
        """"
        compute derivate of lost capital regarding reforestation_investment
        if deforestation_surf < reforestation surf : no dependancies --> derivate is null
        if deforestation_sur > reforestation_surf : derivate is d_delta_reforestation_dinvest * cost_per_ha
        """
        result = d_delta_reforestation_dinvest
        for element in range(0, len(self.years)):
            if abs(self.forest_surface_df.at[element, 'delta_deforestation_surface']) < self.forest_surface_df.at[element, 'delta_reforestation_surface']:
                result[element, element] = 0
            else:
                result[element, element] = result[element,
                                                  element] * self.cost_per_ha
        return result

    def d_lostcapitald_deforestation(self, d_delta_deforestation_d_deforestation):
        """"
        compute derivate of lost capital regarding reforestation_investment
        if deforestation_surf < reforestation surf : d_delta_deforestation_d_deforestation * cost_per_ha
        if deforestation_sur > reforestation_surf : no dependencies
        """
        result = d_delta_deforestation_d_deforestation
        for element in range(0, len(self.years)):
            if abs(self.forest_surface_df.at[element, 'delta_deforestation_surface']) > self.forest_surface_df.at[element, 'delta_reforestation_surface']:
                result[element, element] = 0
            else:
                result[element, element] = result[element,
                                                  element] * self.cost_per_ha

        return result
